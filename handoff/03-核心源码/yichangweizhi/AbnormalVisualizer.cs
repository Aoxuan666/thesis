using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using UnityEngine.EventSystems;

public class AbnormalVisualizer : MonoBehaviour
{
    private const string DefaultTestImageUrl = "d1";
    private static readonly string[] LocalTestImageNames = { "d1", "d2", "d3", "d4", "d5" };

    [System.Serializable]
    public class AbnormalItem
    {
        public int abnormalId;
        public string abnormalType;
        public long inspectionExceptionId;
        public int exceptionTypeId;
        public string exceptionTypeName;
        public string inspectionExceptionTypeName;
        public string exceptionFileName;
        public string exceptionFilePath;
        public string exceptionJudgmentValue;
        public string exceptionPosX;
        public string exceptionPosY;
        public string syncTime;
        public string imageUrl;
        public string fileName;
        public string occurTime;
        public string abnormalTime;
        public float abnormalValue;
        public float mapX;
        public float mapY;
        public int row;
        public int cage;
        public int layer;

        public int InspectionExceptionRowNo;
        public int InspectionExceptionColNo;
        public int InspectionExceptionTierNo;
        public int inspectionExceptionRowNo;
        public int inspectionExceptionColNo;
        public int inspectionExceptionTierNo;
        public int exceptionRowNo;
        public int exceptionColNo;
        public int exceptionTierNo;
    }

    [System.Serializable]
    public class AbnormalResponse
    {
        public List<AbnormalItem> data;
    }

    [System.Serializable]
    public class CurrentInspectionExceptionResponse
    {
        public int code;
        public string msg;
        public CurrentInspectionExceptionData data;
    }

    [System.Serializable]
    public class CurrentInspectionExceptionData
    {
        public List<AbnormalItem> records;
        public long lastId;
    }

    [Header("Cage Generator")]
    public CageHouseGenerator cageHouseGenerator;

    [Header("Number Mapping")]
    public bool useGeneratorDisplayNumbering = true;

    [Header("Marker Prefab Optional")]
    public GameObject markerPrefab;

    [Header("Marker Root")]
    public Transform markerRoot;

    [Header("Default Marker Size")]
    public float markerScale = 0.25f;

    [Header("Panel Position")]
    public float abnormalPanelRightGap = 0.25f;
    [Tooltip("Enable this only in scenes that need a shorter line between the chicken and its information card.")]
    public bool useShortAbnormalConnector = false;
    [InspectorName("Chicken-to-Card Gap (m)")]
    [Range(0.02f, 0.50f)]
    public float abnormalConnectorMaxLength = 0.16f;

    [Header("Play Drop Settings")]
    public bool enableDropPhysics = false;
    public float dropStartHeight = 0.15f;
    public float pinnedHeightOffset = 0.02f;
    public bool freezeXZPosition = true;
    public bool freezeRotation = true;

    [Header("Same Cage Offset")]
    public bool useSameCageOffset = true;
    public Vector2 sameCageRandomRange = new Vector2(0.10f, 0.08f);
    public float sameCageMinDistance = 0.08f;
    public int sameCageRandomTryCount = 100;

    [Header("异常卡片")]
    [Tooltip("关掉后只在笼位上显示异常标记，不生成/不显示旁边那块世界空间信息卡片。")]
    public bool showAbnormalInfoCards = false;

    [Header("Test JSON")]
    public string testImageUrl = DefaultTestImageUrl;

    [TextArea(5, 30)]
    public string testJson =
        "{ \"data\": [" +
        "{ \"abnormalId\": 300001, \"abnormalType\": \"死鸡异常\", \"imageUrl\": \"d1\", \"fileName\": \"d1\", \"abnormalValue\": 0.90, \"mapX\": 1.010, \"mapY\": 1.970, \"row\": 1, \"cage\": 8, \"layer\": 2 }," +
        "{ \"abnormalId\": 300002, \"abnormalType\": \"死鸡异常\", \"imageUrl\": \"d2\", \"fileName\": \"d2\", \"abnormalValue\": 0.94, \"mapX\": 1.020, \"mapY\": 1.980, \"row\": 2, \"cage\": 16, \"layer\": 3 }," +
        "{ \"abnormalId\": 300003, \"abnormalType\": \"死鸡异常\", \"imageUrl\": \"d3\", \"fileName\": \"d3\", \"abnormalValue\": 0.93, \"mapX\": 1.030, \"mapY\": 1.990, \"row\": 4, \"cage\": 25, \"layer\": 1 }," +
        "{ \"abnormalId\": 300004, \"abnormalType\": \"死鸡异常\", \"imageUrl\": \"d4\", \"fileName\": \"d4\", \"abnormalValue\": 0.83, \"mapX\": 1.040, \"mapY\": 2.000, \"row\": 5, \"cage\": 38, \"layer\": 4 }," +
        "{ \"abnormalId\": 300005, \"abnormalType\": \"死鸡异常\", \"imageUrl\": \"d5\", \"fileName\": \"d5\", \"abnormalValue\": 0.81, \"mapX\": 1.050, \"mapY\": 2.010, \"row\": 7, \"cage\": 52, \"layer\": 2 }" +
        "] }";

    private readonly Dictionary<string, List<Vector3>> sameCageOffsetMap =
        new Dictionary<string, List<Vector3>>();

    private readonly List<AbnormalItem> cachedAbnormalItems =
        new List<AbnormalItem>();

    public List<AbnormalItem> GetAbnormalItemsSnapshot()
    {
        return new List<AbnormalItem>(cachedAbnormalItems);
    }

    private bool connectorSettingsInitialized;
    private bool lastUseShortAbnormalConnector;
    private float lastAbnormalConnectorMaxLength;

    private void Awake()
    {
        EnsureMarkerRoot();
        EnsureCageHouseGenerator();
    }

    private void Start()
    {
        // 场景里可能烘焙了旧卡片，这里按开关统一收一次。
        ApplyInfoCardVisibility();
    }

    private void Update()
    {
        if (!connectorSettingsInitialized ||
            lastUseShortAbnormalConnector != useShortAbnormalConnector ||
            !Mathf.Approximately(lastAbnormalConnectorMaxLength, abnormalConnectorMaxLength))
        {
            ApplyConnectorSettingsToExistingPanels();
        }
    }

    private void OnValidate()
    {
        abnormalConnectorMaxLength = Mathf.Clamp(abnormalConnectorMaxLength, 0.02f, 0.50f);
        connectorSettingsInitialized = false;
        ApplyConnectorSettingsToExistingPanels();
        ApplyInfoCardVisibility();
    }

    [ContextMenu("Refresh Abnormal Connector Length")]
    private void ApplyConnectorSettingsToExistingPanels()
    {
        float effectiveLength = useShortAbnormalConnector
            ? Mathf.Clamp(abnormalConnectorMaxLength, 0.02f, 0.50f)
            : 0f;

        // Runtime panels may live under a scene-specific root, so do not limit
        // refreshes to the currently assigned markerRoot.
        AbnormalInfoPanelView[] panels = FindObjectsOfType<AbnormalInfoPanelView>(true);

        for (int i = 0; i < panels.Length; i++)
        {
            if (panels[i] != null)
            {
                panels[i].SetConnectorLengthLimit(effectiveLength);
            }
        }

        lastUseShortAbnormalConnector = useShortAbnormalConnector;
        lastAbnormalConnectorMaxLength = abnormalConnectorMaxLength;
        connectorSettingsInitialized = true;
    }

    /// <summary>把某只异常鸡旁边的信息卡片藏掉（只藏卡片，标记和点击不拿走）。</summary>
    private void HideInfoCardOn(GameObject marker)
    {
        if (marker == null)
        {
            return;
        }

        Transform card = marker.transform.Find("AbnormalInfoPanel");
        if (card != null)
        {
            card.gameObject.SetActive(false);
        }

        AbnormalInfoPanelView view = marker.GetComponent<AbnormalInfoPanelView>();
        if (view != null)
        {
            view.enabled = false;
        }

        // 旧版面板可能叫别的名字、直接挂在标记下，统一按类型再扫一遍。
        AbnormalInfoPanelView[] children = marker.GetComponentsInChildren<AbnormalInfoPanelView>(true);
        for (int i = 0; i < children.Length; i++)
        {
            if (children[i] == null)
            {
                continue;
            }

            children[i].gameObject.SetActive(false);
        }
    }

    /// <summary>按 showAbnormalInfoCards 统一刷新场景里已有的异常卡片（含场景烘焙的旧面板）。</summary>
    [ContextMenu("Apply Abnormal Card Visibility")]
    public void ApplyInfoCardVisibility()
    {
        AbnormalInfoPanelView[] panels = FindObjectsOfType<AbnormalInfoPanelView>(true);
        for (int i = 0; i < panels.Length; i++)
        {
            AbnormalInfoPanelView panel = panels[i];
            if (panel == null)
            {
                continue;
            }

            // 只关卡片本体，不关整只异常鸡，否则标记、点击、统计都没了。
            // 找不到子 Canvas 说明这个面板还没构建，直接跳过，别去关父对象。
            Transform card = panel.transform.Find("AbnormalInfoPanel");
            if (card != null)
            {
                card.gameObject.SetActive(showAbnormalInfoCards);
            }
        }
    }

    [ContextMenu("Test Show Abnormal")]
    public void TestShowAbnormal()
    {
        AbnormalResponse response = JsonUtility.FromJson<AbnormalResponse>(testJson);
        if (response == null || response.data == null || response.data.Count == 0)
        {
            Debug.LogError("Test abnormal JSON has no valid data.");
            return;
        }

        FillMissingLocalImages(response.data);
        ReceiveAbnormalChickenResponse(JsonUtility.ToJson(response));
    }
    [ContextMenu("Clean Duplicate EventSystems")]
    public void CleanDuplicateEventSystems()
    {
        EventSystem[] eventSystems = FindObjectsOfType<EventSystem>(true);
        if (eventSystems == null || eventSystems.Length <= 1)
        {
            Debug.Log("No duplicate EventSystems found.");
            return;
        }

        EventSystem keep = EventSystem.current != null ? EventSystem.current : eventSystems[0];
        int removed = 0;

        for (int i = 0; i < eventSystems.Length; i++)
        {
            EventSystem item = eventSystems[i];
            if (item == null || item == keep)
            {
                continue;
            }

            if (Application.isPlaying)
            {
                Destroy(item.gameObject);
            }
            else
            {
                DestroyImmediate(item.gameObject);
            }

            removed++;
        }

        Debug.Log("Removed duplicate EventSystems: " + removed);
    }

    /// <summary>
    /// Canonical entry point for abnormal-chicken data returned by an API.
    /// Expected JSON shape: { "data": [ ... ] }.
    /// </summary>
    public bool ReceiveAbnormalChickenResponse(string json)
    {
        EnsureMarkerRoot();

        if (string.IsNullOrEmpty(json))
        {
            Debug.LogError("Abnormal JSON is empty.");
            return false;
        }

        List<AbnormalItem> items = TryReadAbnormalItems(json, out long responseLastId);

        if (items == null)
        {
            Debug.LogError("Failed to parse abnormal JSON or data is null.");
            return false;
        }

        cachedAbnormalItems.Clear();
        NormalizeApiFields(items);
        cachedAbnormalItems.AddRange(items);
        FillMissingLocalImages(cachedAbnormalItems);

        RebuildMarkersFromCache();

        Debug.Log($"Abnormal count: {cachedAbnormalItems.Count}, lastId: {responseLastId}");
        return true;
    }

    public bool AppendAbnormalChickenResponse(string json)
    {
        EnsureMarkerRoot();

        if (string.IsNullOrEmpty(json))
        {
            Debug.LogError("Abnormal JSON is empty.");
            return false;
        }

        List<AbnormalItem> items = TryReadAbnormalItems(json, out long responseLastId);
        if (items == null)
        {
            Debug.LogError("Failed to parse abnormal JSON or data is null.");
            return false;
        }

        AddAbnormalItems(items);
        Debug.Log($"Appended abnormal count: {items.Count}, lastId: {responseLastId}");
        return true;
    }

    // Kept for existing callers; all processing is routed through the canonical API entry point.
    public void ShowAbnormalFromJson(string json)
    {
        ReceiveAbnormalChickenResponse(json);
    }

    private static List<AbnormalItem> TryReadAbnormalItems(string json, out long lastId)
    {
        lastId = 0;

        CurrentInspectionExceptionResponse apiResponse =
            JsonUtility.FromJson<CurrentInspectionExceptionResponse>(json);

        if (apiResponse != null && apiResponse.data != null && apiResponse.data.records != null)
        {
            lastId = apiResponse.data.lastId;
            return apiResponse.data.records;
        }

        AbnormalResponse legacyResponse = JsonUtility.FromJson<AbnormalResponse>(json);
        if (legacyResponse != null && legacyResponse.data != null)
        {
            return legacyResponse.data;
        }

        return null;
    }

    /// <summary>
    /// 鍦ㄤ笉娓呯┖缂撳瓨鐨勬儏鍐典笅澧為噺娣诲姞寮傚父銆?
    /// 鍚庣画瀵规帴宸℃鎺ュ彛鏃跺彲鐩存帴璋冪敤銆?
    /// </summary>
    public void AddAbnormalItems(List<AbnormalItem> newItems)
    {
        if (newItems == null || newItems.Count == 0)
        {
            return;
        }

        HashSet<int> existingIds = new HashSet<int>();

        foreach (AbnormalItem item in cachedAbnormalItems)
        {
            existingIds.Add(item.abnormalId);
        }

        int placed = 0;
        int skipped = 0;

        foreach (AbnormalItem item in newItems)
        {
            NormalizeApiFields(item);

            if (existingIds.Contains(item.abnormalId))
            {
                continue;
            }

            FillMissingLocalImage(item, cachedAbnormalItems.Count);
            cachedAbnormalItems.Add(item);

            if (ShowOneAbnormal(item))
            {
                placed++;
            }
            else
            {
                skipped++;
            }
        }

        /*
         * 笼位号缺失（rowNo=-2、colNo=null 这类占位值）时落不到笼位，
         * 这类数据按约定不展示，只在这里记一笔信息日志，
         * 免得被误判成「接口没通」或者「解析失败」。
         */
        if (skipped > 0)
        {
            Debug.Log(
                $"[异常可视化] 本轮 {newItems.Count} 条异常，显示 {placed} 条，" +
                $"未展示 {skipped} 条（数据未带有效笼位号 R/C，无法映射到笼位）。"
            );
        }
        else
        {
            Debug.Log($"[异常可视化] 本轮 {newItems.Count} 条异常，全部展示。");
        }
    }

    /// <summary>
    /// 鐢?CageHouseGenerator 鍦ㄧ偣鍑烩€滃墠鍚庣炕杞?/ 宸﹀彸缈昏浆鈥濆悗鑷姩璋冪敤銆?
    /// 閲嶆柊璇诲彇缂撳瓨寮傚父锛屼娇鏍囪涓庢渶鏂版樉绀虹紪鍙蜂繚鎸佷竴鑷淬€?
    /// </summary>
    public void RefreshMarkersAfterNumberingChanged()
    {
        if (cachedAbnormalItems.Count == 0)
        {
            return;
        }

        RebuildMarkersFromCache();

        Debug.Log("Abnormal markers refreshed after numbering changed.");
    }

    private void RebuildMarkersFromCache()
    {
        EnsureMarkerRoot();
        EnsureCageHouseGenerator();
        ClearMarkerObjectsOnly();
        sameCageOffsetMap.Clear();

        foreach (AbnormalItem item in cachedAbnormalItems)
        {
            ShowOneAbnormal(item);
        }
    }

    private static void NormalizeApiFields(List<AbnormalItem> items)
    {
        if (items == null)
        {
            return;
        }

        foreach (AbnormalItem item in items)
        {
            NormalizeApiFields(item);
        }
    }

    private static void NormalizeApiFields(AbnormalItem item)
    {
        if (item == null)
        {
            return;
        }

        if (string.IsNullOrWhiteSpace(item.abnormalType))
        {
            item.abnormalType = !string.IsNullOrWhiteSpace(item.exceptionTypeName)
                ? item.exceptionTypeName
                : item.inspectionExceptionTypeName;
        }

        if (string.IsNullOrWhiteSpace(item.abnormalType))
        {
            item.abnormalType = GetExceptionTypeName(item.exceptionTypeId, item.exceptionFilePath);
        }

        if (item.abnormalId <= 0 && item.inspectionExceptionId > 0)
        {
            item.abnormalId = item.inspectionExceptionId > int.MaxValue
                ? item.inspectionExceptionId.GetHashCode()
                : (int)item.inspectionExceptionId;
        }

        if (string.IsNullOrWhiteSpace(item.fileName))
        {
            item.fileName = item.exceptionFileName;
        }

        if (string.IsNullOrWhiteSpace(item.imageUrl) &&
            !string.IsNullOrWhiteSpace(item.exceptionFilePath) &&
            !string.IsNullOrWhiteSpace(item.exceptionFileName))
        {
            item.imageUrl = item.exceptionFilePath.TrimEnd('/') + "/" + item.exceptionFileName;
        }

        float parsedValue;
        if (item.abnormalValue <= 0f &&
            float.TryParse(item.exceptionJudgmentValue, NumberStyles.Float, CultureInfo.InvariantCulture, out parsedValue))
        {
            item.abnormalValue = parsedValue;
        }

        float parsedX;
        if (float.TryParse(item.exceptionPosX, NumberStyles.Float, CultureInfo.InvariantCulture, out parsedX))
        {
            item.mapX = parsedX;
        }

        float parsedY;
        if (float.TryParse(item.exceptionPosY, NumberStyles.Float, CultureInfo.InvariantCulture, out parsedY))
        {
            item.mapY = parsedY;
        }

        if (string.IsNullOrWhiteSpace(item.occurTime))
        {
            item.occurTime = item.syncTime;
        }

        // 接口字段与显示编码的对应关系：
        // ColNo -> 列号 C，TierNo -> 层号 T，RowNo -> 组号 G。
        // AbnormalItem 的历史字段命名中 row 保存列号、cage 保存组号。
        int apiGroup = item.InspectionExceptionRowNo > 0
            ? item.InspectionExceptionRowNo
            : item.inspectionExceptionRowNo > 0
                ? item.inspectionExceptionRowNo
                : item.exceptionRowNo;
        int apiColumn = item.InspectionExceptionColNo > 0
            ? item.InspectionExceptionColNo
            : item.inspectionExceptionColNo > 0
                ? item.inspectionExceptionColNo
                : item.exceptionColNo;
        int apiLayer = item.InspectionExceptionTierNo > 0
            ? item.InspectionExceptionTierNo
            : item.inspectionExceptionTierNo > 0
                ? item.inspectionExceptionTierNo
                : item.exceptionTierNo;

        if (item.row <= 0 && apiColumn > 0)
        {
            item.row = apiColumn;
        }

        if (item.cage <= 0 && apiGroup > 0)
        {
            item.cage = apiGroup;
        }

        if (item.layer <= 0 && apiLayer > 0)
        {
            item.layer = apiLayer;
        }
    }

    private static string GetExceptionTypeName(int typeId, string filePath)
    {
        switch (typeId)
        {
            case 1: return "死鸡异常";
            case 4: return "声音异常";
            case 5: return "张嘴异常";
            case 6: return "停产异常";
        }

        string path = filePath ?? string.Empty;
        if (path.Contains("v_deadbroiler")) return "死鸡异常";
        if (path.Contains("w_unhealthybroiler")) return "声音异常";
        if (path.Contains("v_openmouthed")) return "张嘴异常";
        if (path.Contains("v_stoppedlaying")) return "停产异常";
        return "其他异常";
    }

    private static void FillMissingLocalImages(List<AbnormalItem> items)
    {
        if (items == null)
        {
            return;
        }

        for (int i = 0; i < items.Count; i++)
        {
            FillMissingLocalImage(items[i], i);
        }
    }

    private static void FillMissingLocalImage(AbnormalItem item, int index)
    {
        if (item == null || item.inspectionExceptionId > 0 ||
            !string.IsNullOrWhiteSpace(item.imageUrl) || LocalTestImageNames.Length == 0)
        {
            return;
        }

        string imageName = LocalTestImageNames[Mathf.Abs(index) % LocalTestImageNames.Length];
        item.imageUrl = imageName;
        if (string.IsNullOrWhiteSpace(item.fileName))
        {
            item.fileName = imageName;
        }
    }

    private bool ShowOneAbnormal(AbnormalItem item)
    {
        EnsureCageHouseGenerator();
        if (cageHouseGenerator == null)
        {
            Debug.LogError("Please assign CageHouseGenerator first.");
            return false;
        }

        int internalRow = item.row;
        int internalCage = item.cage;
        int internalLayer = item.layer;

        if (useGeneratorDisplayNumbering)
        {
            if (!cageHouseGenerator.TryConvertDisplayToInternal(
                    item.row,
                    item.cage,
                    item.layer,
                    out internalRow,
                    out internalCage,
                    out internalLayer))
            {
                Debug.LogWarning(
                    $"Invalid abnormal display number {item.abnormalId}: " +
                    $"R{item.row:00}-C{item.cage:000}-L{item.layer:00}"
                );
                return false;
            }
        }

        Transform anchor = cageHouseGenerator.GetAnchor(
            internalRow,
            internalCage,
            internalLayer
        );

        if (anchor == null)
        {
            Debug.LogWarning(
                $"Abnormal anchor not found. Display: " +
                $"R{item.row:00}-C{item.cage:000}-L{item.layer:00}, " +
                $"Internal: R{internalRow:00}-C{internalCage:000}-F{internalLayer:00}"
            );
            return false;
        }

        GameObject marker = CreateMarker(
            anchor,
            item,
            internalRow,
            internalCage,
            internalLayer
        );

        marker.name =
            $"Abnormal_{item.abnormalId}_" +
            $"Display_R{item.row:00}_C{item.cage:000}_L{item.layer:00}_" +
            $"Internal_R{internalRow:00}_C{internalCage:000}_F{internalLayer:00}";

        return true;
    }

    private GameObject CreateMarker(
        Transform anchor,
        AbnormalItem item,
        int internalRow,
        int internalCage,
        int internalLayer)
    {
        Vector3 anchorPosition = anchor.position;
        Vector3 sameCageOffset = GetSameCageRandomOffset(
            internalRow,
            internalCage,
            internalLayer,
            item.abnormalId
        );

        Vector3 spawnPosition;

        if (Application.isPlaying && enableDropPhysics)
        {
            spawnPosition = anchorPosition +
                            sameCageOffset +
                            Vector3.up * dropStartHeight;
        }
        else
        {
            spawnPosition = anchorPosition +
                            sameCageOffset +
                            Vector3.up * pinnedHeightOffset;
        }

        GameObject marker;

        if (markerPrefab != null)
        {
            marker = Instantiate(
                markerPrefab,
                spawnPosition,
                Quaternion.identity,
                markerRoot
            );
        }
        else
        {
            marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            marker.transform.position = spawnPosition;
            marker.transform.localScale = Vector3.one * markerScale;

            Renderer renderer = marker.GetComponent<Renderer>();

            if (renderer != null)
            {
                Material mat = new Material(Shader.Find("Standard"));
                mat.color = new Color(0.93f, 0.23f, 0.20f);
                renderer.sharedMaterial = mat;
            }

            if (markerRoot != null)
            {
                marker.transform.SetParent(markerRoot, true);
            }
        }

        FillMarkerInfo(
            marker,
            anchor,
            item,
            internalRow,
            internalCage,
            internalLayer
        );

        if (Application.isPlaying && enableDropPhysics)
        {
            SetupMarkerPhysics(marker);
        }
        else
        {
            SetupPinnedMarker(marker);
        }

        return marker;
    }

    private Vector3 GetSameCageRandomOffset(
        int row,
        int cage,
        int layer,
        int abnormalId)
    {
        if (!useSameCageOffset)
        {
            return Vector3.zero;
        }

        string key = $"{row}_{cage}_{layer}";

        if (!sameCageOffsetMap.TryGetValue(
                key,
                out List<Vector3> usedOffsets))
        {
            usedOffsets = new List<Vector3>();
            sameCageOffsetMap[key] = usedOffsets;
        }

        int index = usedOffsets.Count;

        int seed = abnormalId;
        seed ^= row * 73856093;
        seed ^= cage * 19349663;
        seed ^= layer * 83492791;
        seed ^= index * 265443576;

        System.Random random = new System.Random(seed);

        for (int i = 0; i < sameCageRandomTryCount; i++)
        {
            float x = Mathf.Lerp(
                -sameCageRandomRange.x,
                sameCageRandomRange.x,
                (float)random.NextDouble()
            );

            float z = Mathf.Lerp(
                -sameCageRandomRange.y,
                sameCageRandomRange.y,
                (float)random.NextDouble()
            );

            Vector3 candidate = new Vector3(x, 0f, z);

            if (IsOffsetFarEnough(candidate, usedOffsets))
            {
                usedOffsets.Add(candidate);
                return candidate;
            }
        }

        float fallbackAngle = index * 137.5f * Mathf.Deg2Rad;
        float fallbackRadius = Mathf.Min(
            Mathf.Min(
                sameCageRandomRange.x,
                sameCageRandomRange.y
            ),
            sameCageMinDistance * (1f + index * 0.08f)
        );

        Vector3 fallback = new Vector3(
            Mathf.Cos(fallbackAngle) * fallbackRadius,
            0f,
            Mathf.Sin(fallbackAngle) * fallbackRadius
        );

        usedOffsets.Add(fallback);
        return fallback;
    }

    private bool IsOffsetFarEnough(
        Vector3 candidate,
        List<Vector3> usedOffsets)
    {
        foreach (Vector3 oldOffset in usedOffsets)
        {
            float distance = Vector2.Distance(
                new Vector2(candidate.x, candidate.z),
                new Vector2(oldOffset.x, oldOffset.z)
            );

            if (distance < sameCageMinDistance)
            {
                return false;
            }
        }

        return true;
    }

    private void SetupMarkerPhysics(GameObject marker)
    {
        Collider[] colliders = marker.GetComponentsInChildren<Collider>();

        if (colliders == null || colliders.Length == 0)
        {
            SphereCollider sphereCollider = marker.AddComponent<SphereCollider>();
            sphereCollider.isTrigger = false;
        }
        else
        {
            foreach (Collider col in colliders)
            {
                col.isTrigger = false;
            }
        }

        Rigidbody rb = marker.GetComponent<Rigidbody>();

        if (rb == null)
        {
            rb = marker.AddComponent<Rigidbody>();
        }

        rb.useGravity = true;
        rb.isKinematic = false;

        rb.mass = 1f;
        rb.drag = 0f;
        rb.angularDrag = 0.05f;

        rb.collisionDetectionMode =
            CollisionDetectionMode.ContinuousDynamic;
        rb.interpolation = RigidbodyInterpolation.Interpolate;

        rb.constraints = RigidbodyConstraints.None;

        if (freezeXZPosition)
        {
            rb.constraints |= RigidbodyConstraints.FreezePositionX;
            rb.constraints |= RigidbodyConstraints.FreezePositionZ;
        }

        if (freezeRotation)
        {
            rb.constraints |= RigidbodyConstraints.FreezeRotation;
        }

        rb.WakeUp();
    }

    private void SetupMarkerEditMode(GameObject marker)
    {
        Collider[] colliders = marker.GetComponentsInChildren<Collider>();

        if (colliders == null)
        {
            return;
        }

        foreach (Collider col in colliders)
        {
            col.isTrigger = true;
        }
    }

    private void SetupPinnedMarker(GameObject marker)
    {
        SetupMarkerEditMode(marker);

        Rigidbody rb = marker.GetComponent<Rigidbody>();
        if (rb == null)
        {
            return;
        }

        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        rb.useGravity = false;
        rb.isKinematic = true;
        rb.constraints = RigidbodyConstraints.FreezeAll;
    }

    private void FillMarkerInfo(
        GameObject marker,
        Transform cageAnchor,
        AbnormalItem item,
        int internalRow,
        int internalCage,
        int internalLayer)
    {
        AbnormalMarker markerInfo = marker.GetComponent<AbnormalMarker>();

        if (markerInfo == null)
        {
            markerInfo = marker.AddComponent<AbnormalMarker>();
        }

        markerInfo.abnormalId = item.abnormalId;
        markerInfo.abnormalType = item.abnormalType;
        markerInfo.imageUrl = item.imageUrl;
        markerInfo.fileName = item.fileName;
        markerInfo.occurTime = string.IsNullOrWhiteSpace(item.occurTime) ? "--" : item.occurTime;
        markerInfo.abnormalValue = item.abnormalValue;
        markerInfo.slamX = item.mapX;
        markerInfo.slamY = item.mapY;

        // 杩欓噷淇濈暀鈥滃綋鍓嶅疄闄呭唴閮?Anchor 浣嶇疆鈥濓紝鏂逛究璋冭瘯銆?
        markerInfo.row = internalRow;
        markerInfo.cage = internalCage;
        markerInfo.layer = internalLayer;

        // 卡片关掉时只留异常标记，省掉 BuildIfNeeded 里的整块 Canvas，
        // 异常鸡只本身（红块/模型）照常显示，日志和数据都不受影响。
        if (!showAbnormalInfoCards)
        {
            HideInfoCardOn(marker);
            return;
        }

        AbnormalInfoPanelView panelView = marker.GetComponent<AbnormalInfoPanelView>();
        if (panelView == null)
        {
            panelView = marker.AddComponent<AbnormalInfoPanelView>();
        }

        panelView.maxConnectorLength = useShortAbnormalConnector
            ? Mathf.Clamp(abnormalConnectorMaxLength, 0.02f, 0.50f)
            : 0f;

        string locationCode = AbnormalEventActions.FormatCageCode(item.row, item.layer, item.cage);
        string locationDescription = AbnormalEventActions.FormatChineseCageLocation(item.row, item.layer, item.cage);
        string displayLocation =
            $"笼位：{locationCode}\n" +
            locationDescription;
        markerInfo.locationCode = locationCode;
        markerInfo.locationDescription = locationDescription;
        string displayTime = string.IsNullOrWhiteSpace(item.occurTime) ? "--" : item.occurTime;
        panelView.Initialize(
            item.abnormalType,
            item.abnormalValue,
            displayLocation,
            displayTime,
            item.imageUrl,
            CalculatePanelPositionAtCageRight(marker.transform.position, cageAnchor)
        );

    }

    private Vector3 CalculatePanelPositionAtCageRight(Vector3 chickenPosition, Transform cageAnchor)
    {
        const float panelHalfWidth = 0.86f; // 430 * 0.004 / 2
        Vector3 fallback = chickenPosition + Vector3.right * (panelHalfWidth + abnormalPanelRightGap);
        fallback.y += 0.28f;

        if (cageAnchor == null) return fallback;

        Transform house = cageAnchor;
        while (house.parent != null && !house.name.StartsWith("Column_"))
            house = house.parent;
        if (!house.name.StartsWith("Column_")) return fallback;

        Renderer[] renderers = house.GetComponentsInChildren<Renderer>(true);
        bool found = false;
        float rightBoundaryX = house.InverseTransformPoint(chickenPosition).x;

        foreach (Renderer renderer in renderers)
        {
            if (renderer == null || renderer.GetComponent<TextMesh>() != null) continue;

            Bounds bounds = renderer.bounds;
            Vector3 center = bounds.center;
            Vector3 extents = bounds.extents;
            for (int x = -1; x <= 1; x += 2)
            for (int y = -1; y <= 1; y += 2)
            for (int z = -1; z <= 1; z += 2)
            {
                Vector3 corner = center + Vector3.Scale(extents, new Vector3(x, y, z));
                float localX = house.InverseTransformPoint(corner).x;
                if (!found || localX > rightBoundaryX) rightBoundaryX = localX;
                found = true;
            }
        }

        if (!found) return fallback;

        Vector3 chickenLocal = house.InverseTransformPoint(chickenPosition);
        float chickenToRightBoundary = rightBoundaryX - chickenLocal.x;
        chickenLocal.x += chickenToRightBoundary + panelHalfWidth + abnormalPanelRightGap;
        Vector3 result = house.TransformPoint(chickenLocal);
        result.y = chickenPosition.y + 0.28f;
        return result;
    }

    private void EnsureMarkerRoot()
    {
        if (markerRoot != null && markerRoot.Find("Chicken") == null)
        {
            return;
        }

        if (markerRoot != null)
        {
            Debug.LogWarning("Marker Root points to the marker prefab. RuntimeAbnormalRoot will be used instead.");
            markerRoot = null;
        }

        GameObject root = FindSceneRootObject("RuntimeAbnormalRoot");

        if (root == null)
        {
            root = new GameObject("RuntimeAbnormalRoot");
        }
        else if (!root.activeSelf)
        {
            // 场景里的 RuntimeAbnormalRoot 被关掉过，会导致里面的异常鸡全部不可见。
            Debug.LogWarning(
                "[AbnormalVisualizer] RuntimeAbnormalRoot 处于关闭状态，已自动激活。",
                root
            );
            root.SetActive(true);
        }

        markerRoot = root.transform;
    }

    /// <summary>
    /// 按名字查找场景根节点，包含被禁用的对象。
    /// GameObject.Find 会跳过未激活对象，之前正是这个原因导致场景里已有的
    /// RuntimeAbnormalRoot 找不到，于是又新建了一个空节点，旧标记永远不显示。
    /// </summary>
    private static GameObject FindSceneRootObject(string name)
    {
        UnityEngine.SceneManagement.Scene scene =
            UnityEngine.SceneManagement.SceneManager.GetActiveScene();

        GameObject[] roots = scene.GetRootGameObjects();
        for (int i = 0; i < roots.Length; i++)
        {
            if (roots[i] != null && roots[i].name == name)
            {
                return roots[i];
            }
        }

        return null;
    }

    private void EnsureCageHouseGenerator()
    {
        if (cageHouseGenerator != null)
        {
            return;
        }

        cageHouseGenerator = FindObjectOfType<CageHouseGenerator>(true);
    }

    [ContextMenu("Clear Abnormal Markers")]
    public void ClearMarkers()
    {
        cachedAbnormalItems.Clear();
        sameCageOffsetMap.Clear();
        ClearMarkerObjectsOnly();
    }

    private void ClearMarkerObjectsOnly()
    {
        EnsureMarkerRoot();

        List<GameObject> children = new List<GameObject>();

        for (int i = 0; i < markerRoot.childCount; i++)
        {
            children.Add(markerRoot.GetChild(i).gameObject);
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
    }
}


