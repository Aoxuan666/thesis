using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

/// <summary>
/// 将机器人 SLAM 地图坐标 (mapX, mapY) 映射到 Unity 场景坐标。
/// 标定参数由多个“机器人地图坐标 - Unity 锚点坐标”点对自动拟合得到。
/// </summary>
public class RobotMapLocator : MonoBehaviour
{
    public enum RobotHeadingMode
    {
        ApiAngel,
        MoveDirection,
        None
    }

    [Serializable]
    public class RobotCalibrationPoint
    {
        public string pointName = "Calibration Point";
        public bool useInCalibration = true;

        [Header("Unity 场景中的标定物体")]
        public Transform unityPoint;

        [Header("机器人地图坐标 x / y")]
        public Vector2 mapPoint;
    }

    [Header("动态标定点：至少需要 3 个不共线点，建议使用 4～6 个")]
    public List<RobotCalibrationPoint> calibrationPoints = new List<RobotCalibrationPoint>();

    [Header("机器人根物体")]
    public Transform robotRoot;

    [Header("机器人头顶环境信息")]
    [Tooltip("开启后会自动在机器人头顶生成一个跟随机器人移动的环境信息框。")]
    public bool showEnvironmentInfoPanel = true;

    [Tooltip("机器人信息卡相对机器人模型顶部中心的偏移。Y 越大，卡片越高。")]
    public Vector3 environmentInfoPanelOffset = new Vector3(0f, 0.45f, 0f);

    [Tooltip("机器人信息卡尺寸。X 控制宽度，Y 控制高度。")]
    public Vector2 robotFollowInfoPanelSize = new Vector2(1.25f, 0.74f);

    [Header("仿射变换参数")]
    [Tooltip("UnityX = a * mapX + b * mapY + tx")]
    public float a = 1f;
    public float b = 0f;
    public float tx = 0f;

    [Tooltip("UnityZ = c * mapX + d * mapY + tz")]
    public float c = 0f;
    public float d = 1f;
    public float tz = 0f;

    [Header("Validation Output")]
    [Tooltip("When checking calibration, output a TSV table that can be copied to Excel or Word.")]
    public bool outputValidationTable = true;

    [Header("机器人高度")]
    [Tooltip("机器人根节点在鸡舍地面上的显示高度。")]
    public float groundLocalY = 0.02f;

    [Header("接口更新控制")]
    public bool updatePositionFromApi = true;
    public bool updateRotationFromApi = true;

    [Header("移动平滑")]
    public bool smoothMove = true;

    [Tooltip("接口速度缺失或接近 0 时，到达下一实时点的兜底时长。接口每秒更新一次时建议设为 1。")]
    public float fallbackMoveDuration = 1.0f;

    [Tooltip("目标点与当前显示位置相差超过该值时视为重定位/跳点，直接瞬移到目标。")]
    public float snapDistance = 1.5f;

    [Tooltip("Unity 中允许的最大移动速度，1 Unity 单位=1 m 时单位为 m/s。")]
    public float maxMoveSpeed = 2.0f;

    [Tooltip("机器人接口 Speed 到 Unity 速度的换算。若地图和 Unity 都以米为单位，保持 1。")]
    public float apiSpeedToUnityScale = 1f;

    [Header("朝向设置")]
    [Tooltip("接口已提供 angel 时，推荐选择 ApiAngel。")]
    public RobotHeadingMode headingMode = RobotHeadingMode.ApiAngel;

    [Tooltip("接口 angel 是否与地图坐标系的正方向相反。先保持 false；若机器人左右转向相反，再设为 true。")]
    public bool invertYaw = false;

    [Tooltip("机器人模型车头相对于 Unity Z+ 的修正角度。整体反向填 180；偏左/右 90 度填 ±90。")]
    public float robotYawOffsetDeg = 0f;

    [Tooltip("仅使用 MoveDirection 模式时的模型车头修正。")]
    public float moveDirectionOffsetDeg = 0f;

    [Tooltip("位置变化小于该值时，不根据位移估计朝向。")]
    public float minHeadingMoveDistance = 0.02f;

    [Tooltip("朝向平滑速度。")]
    public float rotationSmoothSpeed = 8.0f;

    [Header("测试放置")]
    public string placePointName = "PatrolStart";

    private Vector3 targetWorldPosition;
    private bool hasTargetPosition = false;
    private float targetMoveSpeed = 0.01f;

    private Vector3 lastWorldPosition;
    private bool hasLastWorldPosition = false;

    private Quaternion targetWorldRotation;
    private bool hasTargetRotation = false;
    private RobotFollowInfoPanel environmentInfoPanel;

    private void Update()
    {
        ResolveRobotRoot();
        EnsureEnvironmentInfoPanel();

        if (robotRoot == null)
        {
            return;
        }

        UpdateRobotPositionSmooth();
        UpdateRobotRotationSmooth();
    }

    private void OnValidate()
    {
        if (environmentInfoPanel != null)
        {
            environmentInfoPanel.worldOffset = GetEffectiveEnvironmentPanelOffset();
            environmentInfoPanel.panelSize = robotFollowInfoPanelSize;
        }
    }

    [ContextMenu("Calculate Affine Calibration")]
    public void CalculateCalibration()
    {
        List<Vector2> mapPoints = new List<Vector2>();
        List<Vector2> unityPoints = new List<Vector2>();

        foreach (RobotCalibrationPoint point in calibrationPoints)
        {
            if (point == null || !point.useInCalibration || point.unityPoint == null)
            {
                continue;
            }

            // 标定锚点统一转换到本脚本所在固定坐标根节点的局部 X-Z 平面。
            Vector3 localPosition = transform.InverseTransformPoint(point.unityPoint.position);

            mapPoints.Add(point.mapPoint);
            unityPoints.Add(new Vector2(localPosition.x, localPosition.z));
        }

        if (mapPoints.Count < 3)
        {
            Debug.LogError("仿射标定至少需要 3 个不共线标定点。");
            return;
        }

        bool success = TrySolveAffine(
            mapPoints,
            unityPoints,
            out float solvedA,
            out float solvedB,
            out float solvedTx,
            out float solvedC,
            out float solvedD,
            out float solvedTz
        );

        if (!success)
        {
            Debug.LogError("仿射标定失败。请检查标定点是否共线、重复，或者地图坐标是否填错。");
            return;
        }

        a = solvedA;
        b = solvedB;
        tx = solvedTx;
        c = solvedC;
        d = solvedD;
        tz = solvedTz;

        Debug.Log(
            "仿射标定完成：\n" +
            "UnityX = " + a.ToString("F6") + " * mapX + " + b.ToString("F6") + " * mapY + " + tx.ToString("F6") + "\n" +
            "UnityZ = " + c.ToString("F6") + " * mapX + " + d.ToString("F6") + " * mapY + " + tz.ToString("F6")
        );

        CheckCalibrationResult();
    }

    [ContextMenu("Check Calibration Result")]
    public void CheckCalibrationResult()
    {
        int count = 0;
        float totalError = 0f;
        float maxError = 0f;
        StringBuilder validationTable = outputValidationTable ? new StringBuilder() : null;

        if (validationTable != null)
        {
            validationTable.AppendLine("Validation point\tSLAM map coordinates (m)\tReference Unity coordinates (m)\tTransformed Unity coordinates (m)\tError (m)");
        }

        foreach (RobotCalibrationPoint point in calibrationPoints)
        {
            if (point == null || !point.useInCalibration || point.unityPoint == null)
            {
                continue;
            }

            Vector3 predictedLocal = MapToUnityLocal(point.mapPoint.x, point.mapPoint.y);
            Vector3 targetLocal = transform.InverseTransformPoint(point.unityPoint.position);

            Vector2 predictedXZ = new Vector2(predictedLocal.x, predictedLocal.z);
            Vector2 targetXZ = new Vector2(targetLocal.x, targetLocal.z);

            float error = Vector2.Distance(predictedXZ, targetXZ);
            totalError += error;
            maxError = Mathf.Max(maxError, error);
            count++;

            string pointName = string.IsNullOrWhiteSpace(point.pointName) ? "P" + count : point.pointName;
            if (validationTable != null)
            {
                validationTable.AppendLine(
                    pointName + "\t" +
                    FormatPoint2(point.mapPoint.x, point.mapPoint.y) + "\t" +
                    FormatPoint2(targetXZ.x, targetXZ.y) + "\t" +
                    FormatPoint2(predictedXZ.x, predictedXZ.y) + "\t" +
                    error.ToString("F3")
                );
            }

            Debug.Log(
                pointName +
                " | Map=" + FormatPoint2(point.mapPoint.x, point.mapPoint.y) +
                " | ReferenceUnity=" + FormatPoint2(targetXZ.x, targetXZ.y) +
                " | TransformedUnity=" + FormatPoint2(predictedXZ.x, predictedXZ.y) +
                " | Plane error = " + error.ToString("F6") + " m"
            );
        }

        if (count > 0)
        {
            float meanError = totalError / count;
            if (validationTable != null)
            {
                validationTable.AppendLine("Mean\t\t\t\t" + meanError.ToString("F3"));
                validationTable.AppendLine("Maximum\t\t\t\t" + maxError.ToString("F3"));
                Debug.Log("Calibration validation table:\n" + validationTable);
            }

            Debug.Log(
                "标定点数量：" + count +
                "，平均误差：" + meanError.ToString("F6") +
                " m，最大误差：" + maxError.ToString("F6") + " m"
            );
        }
    }

    private static string FormatPoint2(float x, float y)
    {
        return "(" + x.ToString("F3") + ", " + y.ToString("F3") + ")";
    }
    /// <summary>
    /// 机器人地图平面坐标转换为本脚本所在坐标根节点的局部坐标。
    /// </summary>
    public Vector3 MapToUnityLocal(float mapX, float mapY)
    {
        float localX = a * mapX + b * mapY + tx;
        float localZ = c * mapX + d * mapY + tz;

        return new Vector3(localX, groundLocalY, localZ);
    }

    /// <summary>
    /// 机器人地图平面坐标转换为 Unity 世界坐标。
    /// </summary>
    public Vector3 MapToUnity(float mapX, float mapY)
    {
        return transform.TransformPoint(MapToUnityLocal(mapX, mapY));
    }

    /// <summary>
    /// 接收接口实时位姿。angelRad 的单位为弧度。
    /// </summary>
    public void UpdateRobotPose(float mapX, float mapY, float angelRad, float speed = 0f)
    {
        UpdateRobotPose(mapX, mapY, angelRad, speed, true);
    }

    /// <summary>
    /// 接收可选航向的实时位姿。接口未提供航向时，applyApiHeading 传 false。
    /// </summary>
    public void UpdateRobotPose(float mapX, float mapY, float angelRad, float speed, bool applyApiHeading)
    {
        if (robotRoot == null)
        {
            Debug.LogWarning("RobotRoot 没有绑定。");
            return;
        }

        Vector3 newWorldPosition = MapToUnity(mapX, mapY);

        if (updatePositionFromApi)
        {
            SetRobotTargetPosition(newWorldPosition, speed);
        }

        if (updateRotationFromApi)
        {
            if (headingMode == RobotHeadingMode.ApiAngel && applyApiHeading)
            {
                targetWorldRotation = MapYawToUnityRotation(angelRad);
                hasTargetRotation = true;
            }
            else if (headingMode == RobotHeadingMode.MoveDirection)
            {
                UpdateMoveDirectionRotation(newWorldPosition);
            }
        }

        lastWorldPosition = newWorldPosition;
        hasLastWorldPosition = true;
    }

    public void UpdateRobotPoseLowLatency(float mapX, float mapY, float angelRad, float interpolationDuration = 0.12f)
    {
        if (robotRoot == null)
        {
            Debug.LogWarning("RobotRoot 没有绑定。");
            return;
        }

        Vector3 newWorldPosition = MapToUnity(mapX, mapY);

        if (updatePositionFromApi)
        {
            float distance = Vector3.Distance(robotRoot.position, newWorldPosition);
            if (!hasTargetPosition || distance > snapDistance)
            {
                robotRoot.position = newWorldPosition;
            }

            targetWorldPosition = newWorldPosition;
            hasTargetPosition = true;
            targetMoveSpeed = Mathf.Max(distance / Mathf.Max(interpolationDuration, 0.02f), 0.01f);
        }

        if (updateRotationFromApi && headingMode == RobotHeadingMode.ApiAngel)
        {
            targetWorldRotation = MapYawToUnityRotation(angelRad);
            hasTargetRotation = true;
        }

        lastWorldPosition = newWorldPosition;
        hasLastWorldPosition = true;
    }

    /// <summary>
    /// 将机器人立即放到接口位置。通常用于首帧对齐、重新定位或充电点对齐。
    /// </summary>
    public void SetRobotPoseInstant(float mapX, float mapY, float angelRad, float speed = 0f)
    {
        bool applyRotation = headingMode == RobotHeadingMode.ApiAngel;
        SetRobotPoseInstant(mapX, mapY, angelRad, applyRotation);
    }

    public void SetRobotPoseInstant(float mapX, float mapY, float angelRad, bool applyRotation)
    {
        if (robotRoot == null)
        {
            Debug.LogWarning("RobotRoot 没有绑定。");
            return;
        }

        Vector3 worldPosition = MapToUnity(mapX, mapY);

        if (updatePositionFromApi)
        {
            robotRoot.position = worldPosition;
            targetWorldPosition = worldPosition;
            hasTargetPosition = true;
            targetMoveSpeed = 0.01f;
        }

        if (applyRotation && updateRotationFromApi)
        {
            robotRoot.rotation = MapYawToUnityRotation(angelRad);
            targetWorldRotation = robotRoot.rotation;
            hasTargetRotation = true;
        }
        else if (headingMode == RobotHeadingMode.MoveDirection && hasLastWorldPosition)
        {
            Vector3 delta = worldPosition - lastWorldPosition;
            delta.y = 0f;

            if (delta.magnitude >= minHeadingMoveDistance)
            {
                Quaternion lookRotation = Quaternion.LookRotation(delta.normalized, Vector3.up);
                Quaternion offsetRotation = Quaternion.Euler(0f, moveDirectionOffsetDeg, 0f);
                robotRoot.rotation = lookRotation * offsetRotation;
                targetWorldRotation = robotRoot.rotation;
                hasTargetRotation = true;
            }
        }

        lastWorldPosition = worldPosition;
        hasLastWorldPosition = true;
    }

    public void SetRobotPoseInstant(double mapX, double mapY, double angelRad, bool applyRotation)
    {
        SetRobotPoseInstant((float)mapX, (float)mapY, (float)angelRad, applyRotation);
    }

    public void SetRobotPoseInstant(double mapX, double mapY, double angelRad, double speed = 0.0)
    {
        SetRobotPoseInstant((float)mapX, (float)mapY, (float)angelRad, (float)speed);
    }

    /// <summary>
    /// 设置机器人下一实时位置目标。
    /// 移动速度取“接口实际速度”和“在 fallbackMoveDuration 内追上目标所需速度”中的较大值，
    /// 以避免每秒一次接口更新时虚拟机器人持续落后于真实机器人。
    /// </summary>
    private void SetRobotTargetPosition(Vector3 newWorldPosition, float speed)
    {
        if (!hasTargetPosition)
        {
            robotRoot.position = newWorldPosition;
            targetWorldPosition = newWorldPosition;
            hasTargetPosition = true;
            targetMoveSpeed = 0.01f;
            return;
        }

        float distance = Vector3.Distance(robotRoot.position, newWorldPosition);

        if (distance > snapDistance)
        {
            robotRoot.position = newWorldPosition;
            targetWorldPosition = newWorldPosition;
            targetMoveSpeed = 0.01f;
            return;
        }

        targetWorldPosition = newWorldPosition;

        float apiMoveSpeed = Mathf.Abs(speed) * apiSpeedToUnityScale;
        float catchUpSpeed = distance / Mathf.Max(fallbackMoveDuration, 0.01f);

        targetMoveSpeed = Mathf.Clamp(
            Mathf.Max(apiMoveSpeed, catchUpSpeed),
            0.01f,
            maxMoveSpeed
        );
    }

    private void UpdateRobotPositionSmooth()
    {
        if (!hasTargetPosition)
        {
            return;
        }

        if (!smoothMove)
        {
            robotRoot.position = targetWorldPosition;
            return;
        }

        float distance = Vector3.Distance(robotRoot.position, targetWorldPosition);

        if (distance < 0.001f)
        {
            robotRoot.position = targetWorldPosition;
            return;
        }

        robotRoot.position = Vector3.MoveTowards(
            robotRoot.position,
            targetWorldPosition,
            targetMoveSpeed * Time.deltaTime
        );
    }

    private void UpdateMoveDirectionRotation(Vector3 newWorldPosition)
    {
        if (!hasLastWorldPosition)
        {
            return;
        }

        Vector3 delta = newWorldPosition - lastWorldPosition;
        delta.y = 0f;

        if (delta.magnitude < minHeadingMoveDistance)
        {
            return;
        }

        Quaternion lookRotation = Quaternion.LookRotation(delta.normalized, Vector3.up);
        Quaternion offsetRotation = Quaternion.Euler(0f, moveDirectionOffsetDeg, 0f);

        targetWorldRotation = lookRotation * offsetRotation;
        hasTargetRotation = true;
    }

    private void UpdateRobotRotationSmooth()
    {
        if (!hasTargetRotation || headingMode == RobotHeadingMode.None)
        {
            return;
        }

        robotRoot.rotation = Quaternion.Slerp(
            robotRoot.rotation,
            targetWorldRotation,
            rotationSmoothSpeed * Time.deltaTime
        );
    }

    /// <summary>
    /// 将接口 angel（弧度）转换为 Unity 世界旋转。
    /// 方向只使用仿射矩阵的线性部分，不使用平移量。
    /// </summary>
    public Quaternion MapYawToUnityRotation(float angelRad)
    {
        float correctedAngelRad = invertYaw ? -angelRad : angelRad;

        Vector2 mapForward = new Vector2(
            Mathf.Cos(correctedAngelRad),
            Mathf.Sin(correctedAngelRad)
        );

        // 与位置映射完全一致的线性变换：
        // [x_u] = [a b] [x_m]
        // [z_u]   [c d] [y_m]
        Vector2 unityForward2D = new Vector2(
            a * mapForward.x + b * mapForward.y,
            c * mapForward.x + d * mapForward.y
        );

        if (unityForward2D.sqrMagnitude < 0.000001f)
        {
            return robotRoot != null ? robotRoot.rotation : Quaternion.identity;
        }

        Vector3 localForward = new Vector3(
            unityForward2D.x,
            0f,
            unityForward2D.y
        ).normalized;

        Quaternion localRotation = Quaternion.LookRotation(localForward, Vector3.up);
        Quaternion modelOffset = Quaternion.Euler(0f, robotYawOffsetDeg, 0f);

        return transform.rotation * localRotation * modelOffset;
    }

    private bool TrySolveAffine(
        List<Vector2> mapPoints,
        List<Vector2> unityPoints,
        out float solvedA,
        out float solvedB,
        out float solvedTx,
        out float solvedC,
        out float solvedD,
        out float solvedTz)
    {
        solvedA = 1f;
        solvedB = 0f;
        solvedTx = 0f;
        solvedC = 0f;
        solvedD = 1f;
        solvedTz = 0f;

        int n = mapPoints.Count;

        float sumX = 0f;
        float sumY = 0f;
        float sumXX = 0f;
        float sumYY = 0f;
        float sumXY = 0f;

        float sumUX = 0f;
        float sumUX_X = 0f;
        float sumUX_Y = 0f;

        float sumUZ = 0f;
        float sumUZ_X = 0f;
        float sumUZ_Y = 0f;

        for (int i = 0; i < n; i++)
        {
            float x = mapPoints[i].x;
            float y = mapPoints[i].y;
            float ux = unityPoints[i].x;
            float uz = unityPoints[i].y;

            sumX += x;
            sumY += y;
            sumXX += x * x;
            sumYY += y * y;
            sumXY += x * y;

            sumUX += ux;
            sumUX_X += ux * x;
            sumUX_Y += ux * y;

            sumUZ += uz;
            sumUZ_X += uz * x;
            sumUZ_Y += uz * y;
        }

        float[,] matrix = new float[3, 3]
        {
            { sumXX, sumXY, sumX },
            { sumXY, sumYY, sumY },
            { sumX,  sumY,  n    }
        };

        float[] rhsX = new float[3]
        {
            sumUX_X,
            sumUX_Y,
            sumUX
        };

        float[] rhsZ = new float[3]
        {
            sumUZ_X,
            sumUZ_Y,
            sumUZ
        };

        if (!Solve3x3(matrix, rhsX, out float[] solutionX) ||
            !Solve3x3(matrix, rhsZ, out float[] solutionZ))
        {
            return false;
        }

        solvedA = solutionX[0];
        solvedB = solutionX[1];
        solvedTx = solutionX[2];
        solvedC = solutionZ[0];
        solvedD = solutionZ[1];
        solvedTz = solutionZ[2];

        return true;
    }

    private bool Solve3x3(float[,] matrix, float[] rhs, out float[] result)
    {
        result = new float[3];
        float[,] augmented = new float[3, 4];

        for (int row = 0; row < 3; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                augmented[row, col] = matrix[row, col];
            }

            augmented[row, 3] = rhs[row];
        }

        for (int pivot = 0; pivot < 3; pivot++)
        {
            int maxRow = pivot;
            float maxAbs = Mathf.Abs(augmented[pivot, pivot]);

            for (int row = pivot + 1; row < 3; row++)
            {
                float currentAbs = Mathf.Abs(augmented[row, pivot]);

                if (currentAbs > maxAbs)
                {
                    maxAbs = currentAbs;
                    maxRow = row;
                }
            }

            if (maxAbs < 0.000001f)
            {
                return false;
            }

            if (maxRow != pivot)
            {
                for (int col = pivot; col < 4; col++)
                {
                    float temp = augmented[pivot, col];
                    augmented[pivot, col] = augmented[maxRow, col];
                    augmented[maxRow, col] = temp;
                }
            }

            float pivotValue = augmented[pivot, pivot];

            for (int col = pivot; col < 4; col++)
            {
                augmented[pivot, col] /= pivotValue;
            }

            for (int row = 0; row < 3; row++)
            {
                if (row == pivot)
                {
                    continue;
                }

                float factor = augmented[row, pivot];

                for (int col = pivot; col < 4; col++)
                {
                    augmented[row, col] -= factor * augmented[pivot, col];
                }
            }
        }

        result[0] = augmented[0, 3];
        result[1] = augmented[1, 3];
        result[2] = augmented[2, 3];

        return true;
    }

    [ContextMenu("Place Robot At First Active Calibration Point")]
    public void PlaceRobotAtFirstActiveCalibrationPoint()
    {
        foreach (RobotCalibrationPoint point in calibrationPoints)
        {
            if (point == null || !point.useInCalibration)
            {
                continue;
            }

            PlaceRobotAtMapPoint(point.mapPoint.x, point.mapPoint.y);
            Debug.Log("机器人已放置到：" + point.pointName);
            return;
        }

        Debug.LogWarning("没有找到启用的标定点。");
    }

    [ContextMenu("Place Robot At Named Calibration Point")]
    public void PlaceRobotAtNamedCalibrationPoint()
    {
        foreach (RobotCalibrationPoint point in calibrationPoints)
        {
            if (point == null || point.pointName != placePointName)
            {
                continue;
            }

            PlaceRobotAtMapPoint(point.mapPoint.x, point.mapPoint.y);
            Debug.Log("机器人已放置到：" + point.pointName);
            return;
        }

        Debug.LogWarning("没有找到名称为 " + placePointName + " 的标定点。");
    }

    public void PlaceRobotAtMapPoint(float mapX, float mapY)
    {
        if (robotRoot == null)
        {
            Debug.LogWarning("RobotRoot 没有绑定。");
            return;
        }

        Vector3 worldPosition = MapToUnity(mapX, mapY);
        robotRoot.position = worldPosition;
        targetWorldPosition = worldPosition;
        hasTargetPosition = true;
        targetMoveSpeed = 0.01f;

        lastWorldPosition = worldPosition;
        hasLastWorldPosition = true;
    }

    public void PlaceRobotAtMapPoint(double mapX, double mapY)
    {
        PlaceRobotAtMapPoint((float)mapX, (float)mapY);
    }

    private void EnsureEnvironmentInfoPanel()
    {
        ResolveRobotRoot();

        if (!showEnvironmentInfoPanel || robotRoot == null)
        {
            return;
        }

        RemoveOldEnvironmentPanels();

        if (environmentInfoPanel == null)
        {
            environmentInfoPanel = robotRoot.GetComponentInChildren<RobotFollowInfoPanel>(true);
        }

        if (environmentInfoPanel == null)
        {
            GameObject panelObject = new GameObject("RobotFollowInfoPanelHost");
            panelObject.transform.SetParent(robotRoot, false);
            environmentInfoPanel = panelObject.AddComponent<RobotFollowInfoPanel>();
        }

        environmentInfoPanel.gameObject.SetActive(true);
        environmentInfoPanel.enabled = true;
        environmentInfoPanel.robotRoot = robotRoot;
        environmentInfoPanel.worldOffset = GetEffectiveEnvironmentPanelOffset();
        environmentInfoPanel.panelSize = robotFollowInfoPanelSize;
    }

    private Vector3 GetEffectiveEnvironmentPanelOffset()
    {
        Vector3 offset = environmentInfoPanelOffset;

        // 兼容旧场景中保存的 1.65 m、2 m、3 m 高度，避免卡片悬在机器人上方过远。
        if (offset.y > 0.75f)
        {
            offset.y = 0.45f;
        }

        return offset;
    }

    private void ResolveRobotRoot()
    {
        if (robotRoot != null)
        {
            return;
        }

        GameObject robotObject = GameObject.Find("RobotRoot");
        if (robotObject != null)
        {
            robotRoot = robotObject.transform;
            return;
        }

        RobotFollowInfoPanel existingPanel = FindObjectOfType<RobotFollowInfoPanel>();
        if (existingPanel != null && existingPanel.robotRoot != null)
        {
            robotRoot = existingPanel.robotRoot;
        }
    }

    private void RemoveOldEnvironmentPanels()
    {
        RobotEnvironmentInfoPanel[] oldPanels = robotRoot.GetComponentsInChildren<RobotEnvironmentInfoPanel>(true);
        for (int i = 0; i < oldPanels.Length; i++)
        {
            if (oldPanels[i] == null)
            {
                continue;
            }

            GameObject panelObject = oldPanels[i].gameObject;
            if (panelObject == robotRoot.gameObject)
            {
                if (Application.isPlaying)
                {
                    Destroy(oldPanels[i]);
                }
                else
                {
                    DestroyImmediate(oldPanels[i]);
                }

                continue;
            }

            if (Application.isPlaying)
            {
                Destroy(panelObject);
            }
            else
            {
                DestroyImmediate(panelObject);
            }
        }
    }
}
