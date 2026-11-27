using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// 参数化鸡舍生成器。
/// 新增：前后/左右编号可独立翻转；翻转后会同步刷新笼位标签和异常标记映射。
/// 注意：flipFrontBackNumbering 与 flipLeftRightNumbering 只改变“编号语义”，不移动鸡笼实体。
/// </summary>
public class CageHouseGenerator : MonoBehaviour
{
    public enum CageDirection
    {
        Left,
        Right
    }

    public enum GenerateAxis
    {
        X,
        Z
    }

    [Header("Four-Layer Cage Prefabs")]
    public GameObject cageColumnPrefabL;
    public GameObject cageColumnPrefabR;

    [Header("Poultry House Layout")]
    [InspectorName("Cage Column Count")]
    public int rowCount = 7;

    [InspectorName("Cage Group Count")]
    public int columnCount = 63;

    [InspectorName("Tier Count")]
    public int layerCount = 4;

    [Header("Generation Direction (Physical Position Only)")]
    [InspectorName("Cage Group Generation Axis")]
    public GenerateAxis columnAxis = GenerateAxis.Z;
    public bool reverseColumnDirection = false;
    public bool reverseRowDirection = false;

    [Header("Cage Group Spacing")]
    [InspectorName("Cage Group Spacing (m)")]
    public float columnSpacing = 1f;

    [Header("Cage Column Center Spacing")]
    [Tooltip("Recommended: control row positions directly using the center-to-center distance between adjacent rows.")]
    public bool useRowCenterSpacings = true;

    [Tooltip("Center-to-center spacing between adjacent rows. Element 0 represents Row 1 to Row 2.")]
    public float[] rowCenterSpacings = new float[]
    {
        3.3f,
        3.3f,
        3.3f,
        3.3f,
        3.3f,
        3.3f
    };

    [Header("Cage Column and Aisle Spacing")]
    public bool useRowGaps = false;

    [Tooltip("Width between the outermost frames of one cage row.")]
    public float rowFrameWidth = 1.8f;

    [Tooltip("Fixed row center-to-center spacing used when Row Gaps is disabled.")]
    public float rowSpacing = 2.8f;

    [Tooltip("Clear gap between adjacent rows. Element 0 represents the gap between Row 1 and Row 2.")]
    public float[] rowGaps = new float[]
    {
        0f,
        1f,
        0f,
        1f,
        0f,
        1f
    };

    [Header("Start Position")]
    public Vector3 startPosition = new Vector3(-19.4f, 0f, 34f);

    [Header("Cage Column Prefab Orientations (Element 0 = Column 1)")]
    [InspectorName("Cage Column Prefab Orientations")]
    public CageDirection[] rowDirections = new CageDirection[]
    {
        CageDirection.Left,
        CageDirection.Right,
        CageDirection.Left,
        CageDirection.Left,
        CageDirection.Right,
        CageDirection.Left,
        CageDirection.Right
    };

    [Header("Anchor Template Naming")]
    public string anchorTemplatePrefix = "AnchorTemplate_F";

    [Header("Cage Number Labels")]
    public bool generateCageNumberLabels = true;

    [Tooltip("Enabled: show a cage number on every tier. Disabled: show it only on Tier 1.")]
    public bool showEveryLayerLabel = true;

    [Tooltip("Treat the bottom tier as Tier 1 when displaying cage numbers.")]
    public bool labelLayerOneIsBottom = true;

    [Tooltip("Treat Unity anchor F01 as the bottom tier.")]
    public bool unityLayerOneIsBottom = true;

    [Tooltip("Local offset of the number label from its anchor. Move it outward to prevent mesh intersection.")]
    public Vector3 labelLocalOffset = new Vector3(0f, 0.12f, -0.15f);

    [Tooltip("Character size of cage number labels.")]
    public float labelCharacterSize = 0.08f;

    [Tooltip("Color of cage number labels.")]
    public Color labelColor = Color.black;

    [Tooltip("Object-name prefix for generated cage labels.")]
    public string cageLabelObjectPrefix = "CageLabel_";

    [Header("Cage Label Camera Facing")]
    [Tooltip("Keep number labels facing the selected camera to prevent mirrored text when viewed from the opposite side.")]
    public bool labelsAlwaysFaceCamera = true;

    [Tooltip("Assign the main camera. If empty, the script automatically finds the camera tagged MainCamera.")]
    public Camera labelTargetCamera;

    [Tooltip("Rotate label facing by 180 degrees if the text still appears mirrored.")]
    public bool invertLabelFacing = false;

    [Header("Dynamic Numbering Direction (Numbering Only)")]
    [Tooltip("Swap the longitudinal numbering direction, such as C001 and C063.")]
    public bool flipFrontBackNumbering = false;

    [Tooltip("Swap the lateral row-numbering direction, such as R01 and R07.")]
    public bool flipLeftRightNumbering = false;

    [Header("Abnormal Marker Synchronization")]
    [Tooltip("Assign the scene AbnormalVisualizer. Displayed abnormalities are remapped after numbering is flipped.")]
    public AbnormalVisualizer abnormalVisualizer;

    [Header("Optional UI Controls")]
    [Tooltip("Assign the front/back numbering flip button.")]
    public Button frontBackFlipButton;

    [Tooltip("Assign the left/right numbering flip button.")]
    public Button leftRightFlipButton;

    [Tooltip("Optional Text component showing the current front/back numbering state.")]
    public Text frontBackStateText;

    [Tooltip("Optional Text component showing the current left/right numbering state.")]
    public Text leftRightStateText;

    [Header("Generation Controls")]
    public bool generateOnStart = true;
    public bool clearOldGeneratedObjects = true;

    [Header("Prefab Save Settings")]
    public string saveFolder = "Assets/Generated";
    public string savePrefabName = "GeneratedChickenHouse.prefab";
    public string savedRootName = "GeneratedChickenHouse";

    [Header("Prefab Orientation Correction")]
    [Tooltip("Rotate each cage prefab when the cage group generation axis is X.")]
    public bool rotatePrefabWhenColumnAxisX = true;

    [Tooltip("Y-axis rotation applied to each cage prefab when the generation axis is X. Typical values are 90 or -90 degrees.")]
    public float prefabRotationYWhenAxisX = 90f;

    private readonly Dictionary<string, Transform> anchorDict =
        new Dictionary<string, Transform>();

    private void Start()
    {
        BindNumberingButtons();
        RefreshNumberingStateText();

        if (!generateCageNumberLabels)
        {
            ClearCageNumberLabelsFromScene();
        }

        if (generateOnStart)
        {
            GenerateHouse();
        }
    }

    private void OnDestroy()
    {
        UnbindNumberingButtons();
    }

    private void BindNumberingButtons()
    {
        if (frontBackFlipButton != null)
        {
            frontBackFlipButton.onClick.RemoveListener(ToggleFrontBackNumbering);
            frontBackFlipButton.onClick.AddListener(ToggleFrontBackNumbering);
        }

        if (leftRightFlipButton != null)
        {
            leftRightFlipButton.onClick.RemoveListener(ToggleLeftRightNumbering);
            leftRightFlipButton.onClick.AddListener(ToggleLeftRightNumbering);
        }
    }

    private void UnbindNumberingButtons()
    {
        if (frontBackFlipButton != null)
        {
            frontBackFlipButton.onClick.RemoveListener(ToggleFrontBackNumbering);
        }

        if (leftRightFlipButton != null)
        {
            leftRightFlipButton.onClick.RemoveListener(ToggleLeftRightNumbering);
        }
    }

    /// <summary>
    /// UI Button 可直接绑定：前后翻转。
    /// C001 与 C063 对调；异常事件按同一规则重新定位。
    /// </summary>
    [ContextMenu("Toggle Front Back Numbering")]
    public void ToggleFrontBackNumbering()
    {
        flipFrontBackNumbering = !flipFrontBackNumbering;
        ApplyNumberingChange("前后编号");
    }

    /// <summary>
    /// UI Button 可直接绑定：左右翻转。
    /// R01 与 R07 对调；异常事件按同一规则重新定位。
    /// </summary>
    [ContextMenu("Toggle Left Right Numbering")]
    public void ToggleLeftRightNumbering()
    {
        flipLeftRightNumbering = !flipLeftRightNumbering;
        ApplyNumberingChange("左右编号");
    }

    [ContextMenu("Reset Numbering Directions")]
    public void ResetNumberingDirections()
    {
        flipFrontBackNumbering = false;
        flipLeftRightNumbering = false;
        ApplyNumberingChange("编号方向已重置");
    }

    private void ApplyNumberingChange(string changeName)
    {
        RegenerateCageNumberLabels();
        RefreshNumberingStateText();

        if (abnormalVisualizer == null)
        {
            abnormalVisualizer = FindObjectOfType<AbnormalVisualizer>();
        }

        if (abnormalVisualizer != null)
        {
            abnormalVisualizer.RefreshMarkersAfterNumberingChanged();
        }

        Debug.Log(
            changeName +
            " 已更新。前后翻转=" + flipFrontBackNumbering +
            "，左右翻转=" + flipLeftRightNumbering
        );
    }

    private void RefreshNumberingStateText()
    {
        if (frontBackStateText != null)
        {
            frontBackStateText.text = flipFrontBackNumbering
                ? "前后编号：已翻转"
                : "前后编号：默认";
        }

        if (leftRightStateText != null)
        {
            leftRightStateText.text = flipLeftRightNumbering
                ? "左右编号：已翻转"
                : "左右编号：默认";
        }
    }

    [ContextMenu("Generate House")]
    public void GenerateHouse()
    {
        if (cageColumnPrefabL == null || cageColumnPrefabR == null)
        {
            Debug.LogError("请先拖入 CageColumn_4Layer_L 和 CageColumn_4Layer_R 两个 Prefab。");
            return;
        }

        NormalizeLayoutArrays();

        if (clearOldGeneratedObjects)
        {
            ClearGeneratedObjects();
        }

        anchorDict.Clear();

        for (int row = 1; row <= rowCount; row++)
        {
            GameObject rowRoot = new GameObject($"Row_{row:00}");
            rowRoot.transform.SetParent(transform);
            rowRoot.transform.localPosition = Vector3.zero;
            rowRoot.transform.localRotation = Quaternion.identity;
            rowRoot.transform.localScale = Vector3.one;

            GameObject prefab = GetPrefabByRow(row);

            for (int col = 1; col <= columnCount; col++)
            {
                Vector3 pos = GetColumnPosition(row, col);

                Quaternion columnRotation = Quaternion.identity;

                if (columnAxis == GenerateAxis.X && rotatePrefabWhenColumnAxisX)
                {
                    columnRotation = Quaternion.Euler(
                        0f,
                        prefabRotationYWhenAxisX,
                        0f
                    );
                }

                GameObject columnObj = Instantiate(
                    prefab,
                    pos,
                    columnRotation,
                    rowRoot.transform
                );

                // 这里的名称始终是“内部物理编号”，不要手动改。
                columnObj.name = $"Column_R{row:00}_C{col:000}";

                RegisterAnchors(columnObj.transform, row, col);
            }
        }

        RefreshNumberingStateText();

        Debug.Log(
            $"鸡笼生成完成：{rowCount}排 × {columnCount}笼 × {layerCount}层，" +
            $"Anchor数量：{anchorDict.Count}"
        );
    }

    [ContextMenu("Clear Generated Objects")]
    public void ClearGeneratedObjects()
    {
        List<GameObject> children = new List<GameObject>();

        for (int i = 0; i < transform.childCount; i++)
        {
            children.Add(transform.GetChild(i).gameObject);
        }

        foreach (GameObject child in children)
        {
            if (Application.isPlaying)
            {
                Destroy(child);
            }
            else
            {
                DestroyImmediate(child);
            }
        }

        anchorDict.Clear();

        Debug.Log("已清除生成的鸡舍对象。");
    }

    [ContextMenu("Apply Template Layout Defaults")]
    public void ApplyTemplateLayoutDefaults()
    {
        columnAxis = GenerateAxis.Z;
        reverseColumnDirection = false;
        reverseRowDirection = false;
        columnSpacing = 1f;

        useRowCenterSpacings = true;
        rowCenterSpacings = new float[]
        {
            3.3f,
            3.3f,
            3.3f,
            3.3f,
            3.3f,
            3.3f
        };

        useRowGaps = false;
        rowFrameWidth = 1.8f;
        rowSpacing = 3.3f;
        rowGaps = new float[]
        {
            0f,
            1f,
            0f,
            1f,
            0f,
            1f
        };

        startPosition = new Vector3(-19.4f, 0f, 34f);
        rowDirections = new CageDirection[]
        {
            CageDirection.Left,
            CageDirection.Right,
            CageDirection.Left,
            CageDirection.Left,
            CageDirection.Right,
            CageDirection.Left,
            CageDirection.Right
        };

        labelLayerOneIsBottom = false;
        unityLayerOneIsBottom = false;

        Debug.Log("已套用模板布局参数。请重新执行 Generate House。");
    }

    [ContextMenu("Save Generated House As Prefab")]
    public void SaveGeneratedHouseAsPrefab()
    {
#if UNITY_EDITOR
        if (transform.childCount == 0)
        {
            Debug.LogError("当前没有生成任何鸡舍对象，无法保存 Prefab。请先执行 Generate House。");
            return;
        }

        GameObject tempRoot = new GameObject(savedRootName);

        tempRoot.transform.position = transform.position;
        tempRoot.transform.rotation = transform.rotation;
        tempRoot.transform.localScale = transform.localScale;

        List<Transform> generatedRows = new List<Transform>();

        for (int i = 0; i < transform.childCount; i++)
        {
            Transform child = transform.GetChild(i);

            if (child.name.StartsWith("Row_"))
            {
                generatedRows.Add(child);
            }
        }

        if (generatedRows.Count == 0)
        {
            DestroyImmediate(tempRoot);
            Debug.LogError("没有找到 Row_ 开头的生成排对象，无法保存 Prefab。");
            return;
        }

        foreach (Transform row in generatedRows)
        {
            GameObject rowCopy = Instantiate(row.gameObject, tempRoot.transform);
            rowCopy.name = row.name;

            rowCopy.transform.position = row.position;
            rowCopy.transform.rotation = row.rotation;
            rowCopy.transform.localScale = row.localScale;
        }

        EnsureFolderExists(saveFolder);

        string prefabPath = saveFolder + "/" + savePrefabName;

        PrefabUtility.SaveAsPrefabAsset(tempRoot, prefabPath);

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        DestroyImmediate(tempRoot);

        Debug.Log("鸡舍 Prefab 已保存到：" + prefabPath);
#else
        Debug.LogWarning("保存 Prefab 只能在 Unity 编辑器中使用，打包运行后不可用。");
#endif
    }

#if UNITY_EDITOR
    private void EnsureFolderExists(string folderPath)
    {
        if (AssetDatabase.IsValidFolder(folderPath))
        {
            return;
        }

        string[] folders = folderPath.Split('/');

        if (folders.Length == 0 || folders[0] != "Assets")
        {
            Debug.LogError("保存路径必须以 Assets 开头，例如：Assets/Generated");
            return;
        }

        string currentPath = "Assets";

        for (int i = 1; i < folders.Length; i++)
        {
            string nextPath = currentPath + "/" + folders[i];

            if (!AssetDatabase.IsValidFolder(nextPath))
            {
                AssetDatabase.CreateFolder(currentPath, folders[i]);
            }

            currentPath = nextPath;
        }
    }
#endif

    private GameObject GetPrefabByRow(int row)
    {
        int index = row - 1;

        if (rowDirections == null || index >= rowDirections.Length)
        {
            Debug.LogWarning($"第 {row} 排没有配置方向，默认使用 Right。");
            return cageColumnPrefabR;
        }

        return rowDirections[index] == CageDirection.Left
            ? cageColumnPrefabL
            : cageColumnPrefabR;
    }

    private Vector3 GetColumnPosition(int row, int col)
    {
        float colOffset = (col - 1) * columnSpacing;
        float rowOffset = GetRowOffset(row);

        if (reverseColumnDirection)
        {
            colOffset = -colOffset;
        }

        if (reverseRowDirection)
        {
            rowOffset = -rowOffset;
        }

        Vector3 pos = startPosition;

        if (columnAxis == GenerateAxis.X)
        {
            pos.x += colOffset;
            pos.z += rowOffset;
        }
        else
        {
            pos.z += colOffset;
            pos.x += rowOffset;
        }

        return pos;
    }

    private float GetRowOffset(int row)
    {
        if (row <= 1)
        {
            return 0f;
        }

        if (useRowCenterSpacings)
        {
            float offset = 0f;

            for (int i = 0; i < row - 1; i++)
            {
                float spacing = rowSpacing;

                if (rowCenterSpacings != null && i < rowCenterSpacings.Length)
                {
                    spacing = rowCenterSpacings[i];
                }
                else
                {
                    Debug.LogWarning(
                        $"rowCenterSpacings 缺少第 {i + 1} 个中心距，默认使用 rowSpacing={rowSpacing}。"
                    );
                }

                offset += spacing;
            }

            return offset;
        }

        if (!useRowGaps)
        {
            return (row - 1) * rowSpacing;
        }

        float legacyOffset = 0f;

        for (int i = 0; i < row - 1; i++)
        {
            float gap = 0f;

            if (rowGaps != null && i < rowGaps.Length)
            {
                gap = rowGaps[i];
            }
            else
            {
                Debug.LogWarning(
                    $"rowGaps 缺少第 {i + 1} 个间隔，默认按 0 米处理。"
                );
            }

            legacyOffset += rowFrameWidth + gap;
        }

        return legacyOffset;
    }

    private void NormalizeLayoutArrays()
    {
        int spacingCount = Mathf.Max(0, rowCount - 1);

        if (rowCenterSpacings == null || rowCenterSpacings.Length < spacingCount)
        {
            float[] normalized = new float[spacingCount];

            for (int i = 0; i < spacingCount; i++)
            {
                normalized[i] =
                    rowCenterSpacings != null && i < rowCenterSpacings.Length
                        ? rowCenterSpacings[i]
                        : rowSpacing;
            }

            rowCenterSpacings = normalized;
        }

        if (rowDirections == null || rowDirections.Length < rowCount)
        {
            CageDirection[] templateDirections = new CageDirection[]
            {
                CageDirection.Left,
                CageDirection.Right,
                CageDirection.Left,
                CageDirection.Left,
                CageDirection.Right,
                CageDirection.Left,
                CageDirection.Right
            };

            CageDirection[] normalized = new CageDirection[rowCount];

            for (int i = 0; i < rowCount; i++)
            {
                if (rowDirections != null && i < rowDirections.Length)
                {
                    normalized[i] = rowDirections[i];
                }
                else if (i < templateDirections.Length)
                {
                    normalized[i] = templateDirections[i];
                }
                else
                {
                    normalized[i] = CageDirection.Right;
                }
            }

            rowDirections = normalized;
        }
    }

    private void RegisterAnchors(Transform columnTransform, int row, int col)
    {
        for (int layer = 1; layer <= layerCount; layer++)
        {
            string key = GetAnchorKey(row, col, layer);
            string templateName = $"{anchorTemplatePrefix}{layer:00}";

            Transform anchor = FindChildRecursive(columnTransform, templateName);

            if (anchor == null)
            {
                anchor = FindChildRecursive(columnTransform, key);
            }

            if (anchor == null)
            {
                Debug.LogWarning($"{columnTransform.name} 缺少 {templateName}");
                continue;
            }

            // Anchor 以“内部物理编号”保存。标签和异常数据通过转换函数映射。
            anchor.name = key;
            anchorDict[key] = anchor;

            CreateCageNumberLabel(anchor, row, col, layer);
        }
    }

    private void CreateCageNumberLabel(
        Transform anchor,
        int internalRow,
        int internalCage,
        int internalLayer)
    {
        if (!generateCageNumberLabels)
        {
            return;
        }

        int displayRow = ConvertInternalRowToDisplayRow(internalRow);
        int displayCage = ConvertInternalCageToDisplayCage(internalCage);
        int displayLayer = ConvertInternalLayerToDisplayLayer(internalLayer);

        if (!showEveryLayerLabel && displayLayer != 1)
        {
            return;
        }

        string labelText = showEveryLayerLabel
            ? $"R{displayRow:00}-C{displayCage:000}-L{displayLayer:00}"
            : $"R{displayRow:00}-C{displayCage:000}";

        GameObject labelObj = new GameObject(cageLabelObjectPrefix + labelText);
        labelObj.transform.SetParent(anchor, false);
        labelObj.transform.localPosition = labelLocalOffset;
        labelObj.transform.localRotation = Quaternion.identity;
        labelObj.transform.localScale = Vector3.one;

        TextMesh textMesh = labelObj.AddComponent<TextMesh>();
        textMesh.text = labelText;
        textMesh.characterSize = labelCharacterSize;
        textMesh.anchor = TextAnchor.MiddleCenter;
        textMesh.alignment = TextAlignment.Center;
        textMesh.color = labelColor;

        MeshRenderer meshRenderer = labelObj.GetComponent<MeshRenderer>();

        if (meshRenderer != null)
        {
            meshRenderer.shadowCastingMode =
                UnityEngine.Rendering.ShadowCastingMode.Off;
            meshRenderer.receiveShadows = false;
        }

        // 文字作为三维标签时，若从背面观看会镜像。
        // 这里让标签自动面向相机；编号前后/左右翻转后重新生成标签时，
        // 新标签也会沿用相同视角规则。
        if (labelsAlwaysFaceCamera)
        {
            CageNumberLabelBillboard billboard =
                labelObj.AddComponent<CageNumberLabelBillboard>();
            billboard.targetCamera = labelTargetCamera;
            billboard.invertFacing = invertLabelFacing;
        }
    }

    // =========================
    // 编号映射：显示编号 <-> 内部物理编号
    // =========================

    public int ConvertDisplayRowToInternalRow(int displayRow)
    {
        return flipLeftRightNumbering
            ? rowCount - displayRow + 1
            : displayRow;
    }

    public int ConvertDisplayCageToInternalCage(int displayCage)
    {
        return flipFrontBackNumbering
            ? columnCount - displayCage + 1
            : displayCage;
    }

    public int ConvertDisplayLayerToInternalLayer(int displayLayer)
    {
        return labelLayerOneIsBottom == unityLayerOneIsBottom
            ? displayLayer
            : layerCount - displayLayer + 1;
    }

    public int ConvertInternalRowToDisplayRow(int internalRow)
    {
        // 左右翻转是自反映射，公式与 Display->Internal 相同。
        return flipLeftRightNumbering
            ? rowCount - internalRow + 1
            : internalRow;
    }

    public int ConvertInternalCageToDisplayCage(int internalCage)
    {
        // 前后翻转是自反映射，公式与 Display->Internal 相同。
        return flipFrontBackNumbering
            ? columnCount - internalCage + 1
            : internalCage;
    }

    public int ConvertInternalLayerToDisplayLayer(int internalLayer)
    {
        return labelLayerOneIsBottom == unityLayerOneIsBottom
            ? internalLayer
            : layerCount - internalLayer + 1;
    }

    public bool TryConvertDisplayToInternal(
        int displayRow,
        int displayCage,
        int displayLayer,
        out int internalRow,
        out int internalCage,
        out int internalLayer)
    {
        internalRow = ConvertDisplayRowToInternalRow(displayRow);
        internalCage = ConvertDisplayCageToInternalCage(displayCage);
        internalLayer = ConvertDisplayLayerToInternalLayer(displayLayer);

        bool valid =
            internalRow >= 1 && internalRow <= rowCount &&
            internalCage >= 1 && internalCage <= columnCount &&
            internalLayer >= 1 && internalLayer <= layerCount;

        if (!valid)
        {
            Debug.LogWarning(
                $"显示编号超出范围：R{displayRow:00}-C{displayCage:000}-L{displayLayer:00}，" +
                $"转换后内部编号：R{internalRow:00}-C{internalCage:000}-F{internalLayer:00}"
            );
        }

        return valid;
    }

    /// <summary>
    /// 用“当前场景显示的 R/C/L 编号”查询 Anchor。
    /// 异常可视化必须使用这个方法，才会与两个翻转按钮同步。
    /// </summary>
    public Transform GetAnchorByDisplayNumber(
        int displayRow,
        int displayCage,
        int displayLayer)
    {
        if (!TryConvertDisplayToInternal(
                displayRow,
                displayCage,
                displayLayer,
                out int internalRow,
                out int internalCage,
                out int internalLayer))
        {
            return null;
        }

        return GetAnchor(internalRow, internalCage, internalLayer);
    }

    [ContextMenu("Rebuild Anchors From Scene")]
    public void RebuildAnchorsFromScene()
    {
        anchorDict.Clear();

        int successCount = 0;

        for (int row = 1; row <= rowCount; row++)
        {
            Transform rowRoot = transform.Find($"Row_{row:00}");

            if (rowRoot == null)
            {
                Debug.LogWarning($"未找到 Row_{row:00}");
                continue;
            }

            for (int col = 1; col <= columnCount; col++)
            {
                Transform column = rowRoot.Find($"Column_R{row:00}_C{col:000}");

                if (column == null)
                {
                    Debug.LogWarning($"未找到 Column_R{row:00}_C{col:000}");
                    continue;
                }

                for (int layer = 1; layer <= layerCount; layer++)
                {
                    string key = GetAnchorKey(row, col, layer);
                    string templateName = $"{anchorTemplatePrefix}{layer:00}";

                    Transform anchor = FindChildRecursive(column, key);

                    if (anchor == null)
                    {
                        anchor = FindChildRecursive(column, templateName);
                    }

                    if (anchor == null)
                    {
                        Debug.LogWarning(
                            $"{column.name} 未找到第 {layer} 层 Anchor，" +
                            $"目标名称：{key} 或 {templateName}"
                        );
                        continue;
                    }

                    anchor.name = key;
                    anchorDict[key] = anchor;
                    successCount++;
                }
            }
        }

        Debug.Log($"Anchor 字典重建完成，数量：{successCount}");
    }

    [ContextMenu("Regenerate Cage Number Labels")]
    public void RegenerateCageNumberLabels()
    {
        ClearCageNumberLabelsFromScene();

        if (!generateCageNumberLabels)
        {
            Debug.Log("generateCageNumberLabels 未开启，只清除了已有编号。");
            return;
        }

        if (anchorDict.Count == 0)
        {
            RebuildAnchorsFromScene();
        }

        int labelCount = 0;

        for (int row = 1; row <= rowCount; row++)
        {
            for (int col = 1; col <= columnCount; col++)
            {
                for (int layer = 1; layer <= layerCount; layer++)
                {
                    string key = GetAnchorKey(row, col, layer);

                    if (anchorDict.TryGetValue(key, out Transform anchor))
                    {
                        int displayLayer =
                            ConvertInternalLayerToDisplayLayer(layer);

                        if (!showEveryLayerLabel && displayLayer != 1)
                        {
                            continue;
                        }

                        CreateCageNumberLabel(anchor, row, col, layer);
                        labelCount++;
                    }
                }
            }
        }

        RefreshNumberingStateText();

        Debug.Log($"笼子编号重新生成完成，数量：{labelCount}");
    }

    [ContextMenu("Clear Cage Number Labels")]
    public void ClearCageNumberLabelsFromScene()
    {
        List<GameObject> labels = new List<GameObject>();

        CollectLabelObjects(transform, labels);

        foreach (GameObject label in labels)
        {
            if (Application.isPlaying)
            {
                Destroy(label);
            }
            else
            {
                DestroyImmediate(label);
            }
        }

        Debug.Log($"已清除笼子编号，数量：{labels.Count}");
    }

    private void CollectLabelObjects(Transform parent, List<GameObject> labels)
    {
        foreach (Transform child in parent)
        {
            if (child.name.StartsWith(cageLabelObjectPrefix))
            {
                labels.Add(child.gameObject);
            }
            else
            {
                CollectLabelObjects(child, labels);
            }
        }
    }

    private Transform FindChildRecursive(Transform parent, string targetName)
    {
        foreach (Transform child in parent)
        {
            if (child.name == targetName)
            {
                return child;
            }

            Transform result = FindChildRecursive(child, targetName);

            if (result != null)
            {
                return result;
            }
        }

        return null;
    }

    /// <summary>
    /// 内部物理编号查询 Anchor。外部异常数据不要直接调用，改用 GetAnchorByDisplayNumber。
    /// </summary>
    public Transform GetAnchor(int row, int col, int layer)
    {
        string key = GetAnchorKey(row, col, layer);

        if (anchorDict.TryGetValue(key, out Transform anchor))
        {
            return anchor;
        }

        RebuildAnchorsFromScene();

        if (anchorDict.TryGetValue(key, out anchor))
        {
            return anchor;
        }

        Debug.LogWarning("未找到定位点：" + key);
        return null;
    }

    public string GetAnchorKey(int row, int col, int layer)
    {
        return $"Anchor_R{row:00}_C{col:000}_F{layer:00}";
    }

    public bool HasAnchor(int row, int col, int layer)
    {
        string key = GetAnchorKey(row, col, layer);
        return anchorDict.ContainsKey(key);
    }

    public int GetAnchorCount()
    {
        return anchorDict.Count;
    }
}
