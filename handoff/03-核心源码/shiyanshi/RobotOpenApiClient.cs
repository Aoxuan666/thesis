using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

/// <summary>
/// 机器人开放接口客户端。
/// 每隔 interval 秒请求实时状态，并交给 RobotMapLocator 映射为 Unity 中的实时位姿。
/// </summary>
public class RobotOpenApiClient : MonoBehaviour
{
    private const string PublicApiBaseUrl = "http://{host}:{port}";
    private const string LocalApiBaseUrl = "http://{host}:{port}";
    private const string StatusPath = "/open/robot/status";
    private const string StatusListPath = "/open/robot/status/list";
    private const string EmptyBodySha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    [Header("API")]
    [Tooltip("开放接口根地址。可填公网或内网地址；旧接口地址会在运行时自动迁移。")]
    public string apiUrl = PublicApiBaseUrl;

    public string appKey = "{APP_KEY}";

    [Tooltip("HMAC 签名密钥。WebGL 构建无法安全保存密钥，生产环境应通过服务端代理请求。")]
    public string appSecret = "{APP_SECRET}";

    [Tooltip("只处理该机器人 ID 的消息。留空表示接收任意机器人。")]
    public string targetRobotId = "B{ROBOT_IMEI_C}";

    [Header("栋舍绑定")]
    [Tooltip("机器人状态里的 houseCode 变化时，自动切换巡检栋舍并重新拉取该栋舍的巡检数据。")]
    public bool syncInspectionHouseOnRobotSwitch = true;

    [Header("轮询设置")]
    public bool autoStart = true;

    [Tooltip("机器人状态轮询间隔（秒）。当前场景使用 0.6 秒。")]
    public float interval = 0.6f;

    [Header("机器人定位")]
    public RobotMapLocator locator;

    [Tooltip("优先使用机器人本机 SLAM 位姿接口更新虚拟场景，后端状态接口不再覆盖坐标。")]
    public bool useRobotLocalPose = false;

    public bool updateLocatorFromStatusApi = true;

    [Header("UI 显示")]
    [Tooltip("是否根据接口数据刷新 UI 面板。只影响 UI，不影响机器人定位逻辑。")]
    public bool updateUI = true;

    [Tooltip("左侧机器人状态普通 Text 文字")]
    public Text robotStatusText;

    [Tooltip("右侧环境参数普通 Text 文字")]
    public Text environmentStatusText;

    [Tooltip("新版数字孪生 HUD。未手动绑定时会在收到接口数据后自动查找。")]
    public PoultryTwinHUDController poultryTwinHUD;

    [Header("启动控制")]
    [Tooltip("第一帧接口数据只用于瞬间对齐机器人位置和朝向，不播放移动动画。")]
    public bool firstStatusOnlyAlign = true;

    [Tooltip("机器人 ChargeStatus=true 时，是否瞬间对齐位置和朝向。")]
    public bool instantAlignWhenCharging = true;

    [Header("调试")]
    public bool logRawResponse = false;
    public bool logParsedData = false;
    public bool logRuntimeState = true;

    private Coroutine pollingCoroutine;

    /// <summary>上一次生效的栋舍。用 long.MinValue 表示「还没解析过」，保证第一次状态一定触发切换。</summary>
    private long currentHouseCode = long.MinValue;
    private AbnormalChickenApiClient abnormalChickenApiClient;
    private bool hasReceivedFirstStatus = false;
    private bool hasWarnedAboutLegacyUrl = false;

    /// <summary>被限流（429）时把间隔按倍数放宽，成功后慢慢收回。</summary>
    private float rateLimitBackoff = 1f;
    private bool hasWarnedRateLimit = false;

    private int lastLiftPosition = -1;
    private bool hasLastLiftPosition = false;

    private RobotRuntimeState currentRuntimeState = RobotRuntimeState.Unknown;
    private readonly List<string> discoveredRobotIds = new List<string>();

    /*
     * HUD 下拉列表用的可读标签（机器人名 + 栋舍）。
     * robotId 是 15 位设备号，下拉里全是数字根本分不清是哪一台，
     * 所以额外记一份标签；真正切换仍然按 robotId 走，标签只影响显示。
     */
    private readonly Dictionary<string, string> discoveredRobotLabels =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

    public IList<string> DiscoveredRobotIds => discoveredRobotIds;

    /// <summary>取可读标签；还没拿到名称时退回 robotId。</summary>
    public string GetDiscoveredRobotLabel(string robotId)
    {
        if (string.IsNullOrWhiteSpace(robotId))
        {
            return string.Empty;
        }

        string normalized = robotId.Trim();
        string label;

        if (discoveredRobotLabels.TryGetValue(normalized, out label) &&
            !string.IsNullOrWhiteSpace(label))
        {
            return label;
        }

        return normalized;
    }

    private void Start()
    {
        RobotLocalPoseClient existingLocalPoseClient = GetComponent<RobotLocalPoseClient>();

        if (useRobotLocalPose)
        {
            RobotLocalPoseClient localPoseClient = GetComponent<RobotLocalPoseClient>();
            if (localPoseClient == null)
            {
                localPoseClient = gameObject.AddComponent<RobotLocalPoseClient>();
            }
            localPoseClient.enabled = true;
            localPoseClient.Configure(locator);
        }
        else if (existingLocalPoseClient != null)
        {
            existingLocalPoseClient.enabled = false;
        }

        if (autoStart)
        {
            StartPolling();
        }
    }

    [ContextMenu("Start Polling")]
    public void StartPolling()
    {
        StopPolling();
        pollingCoroutine = StartCoroutine(PollingLoop());
    }

    [ContextMenu("Stop Polling")]
    public void StopPolling()
    {
        if (pollingCoroutine != null)
        {
            StopCoroutine(pollingCoroutine);
            pollingCoroutine = null;
        }
    }

    [ContextMenu("Request Once")]
    public void RequestOnce()
    {
        StartCoroutine(RequestRobotStatus());
    }

    [ContextMenu("Reset Runtime State")]
    public void ResetRuntimeState()
    {
        hasReceivedFirstStatus = false;
        hasLastLiftPosition = false;
        lastLiftPosition = -1;
        currentRuntimeState = RobotRuntimeState.Unknown;

        // 换机器人后必须重新解析栋舍，否则 houseCode 没变就会跳过切换。
        currentHouseCode = long.MinValue;

        Debug.Log("RobotOpenApiClient 运行状态已重置。");
    }

    private IEnumerator PollingLoop()
    {
        while (true)
        {
            yield return RequestRobotStatus();
            yield return new WaitForSeconds(GetEffectiveInterval());
        }
    }

    private IEnumerator RequestRobotStatus()
    {
        if (string.IsNullOrWhiteSpace(appKey) || string.IsNullOrWhiteSpace(appSecret))
        {
            Debug.LogError("机器人开放接口 appKey 或 appSecret 未配置。");
            yield break;
        }

        bool requestList = string.IsNullOrWhiteSpace(targetRobotId);
        string path = requestList ? StatusListPath : StatusPath;
        SortedDictionary<string, string> parameters = new SortedDictionary<string, string>(StringComparer.Ordinal)
        {
            { "detail", "full" }
        };

        if (requestList)
        {
            parameters.Add("onlineOnly", "false");
        }
        else
        {
            parameters.Add("robotId", targetRobotId.Trim());
        }

        using (UnityWebRequest request = CreateSignedGetRequest(path, parameters))
        {
            if (request == null)
            {
                yield break;
            }

            request.timeout = 10;
            yield return request.SendWebRequest();

            string responseJson = request.downloadHandler == null ? string.Empty : request.downloadHandler.text;

            if (request.result != UnityWebRequest.Result.Success)
            {
                /*
                 * 429 是平台限流，不是接口坏了：放宽间隔、只提示一次，
                 * 别把控制台刷成红色。
                 */
                if (request.responseCode == 429L)
                {
                    HandleRateLimited();
                }
                else
                {
                    Debug.LogWarning(
                        "机器人接口请求失败：" + request.error +
                        "，HTTP 状态码：" + request.responseCode +
                        (string.IsNullOrWhiteSpace(responseJson) ? string.Empty : "\n返回：" + responseJson)
                    );
                }

                yield break;
            }

            /* 成功就收一档退避，直到回到基准间隔。 */
            rateLimitBackoff = Mathf.Max(1f, rateLimitBackoff * 0.5f);

            if (rateLimitBackoff <= 1f)
            {
                hasWarnedRateLimit = false;
            }

            if (logRawResponse)
            {
                Debug.Log("接口原始返回：\n" + responseJson);
            }

            if (string.IsNullOrWhiteSpace(responseJson))
            {
                Debug.LogWarning("接口返回为空。");
                yield break;
            }

            if (requestList)
            {
                HandleStatusListResponse(responseJson);
            }
            else
            {
                HandleSingleStatusResponse(responseJson);
            }
        }
    }

    private UnityWebRequest CreateSignedGetRequest(string path, SortedDictionary<string, string> parameters)
    {
        string timestamp = GetUnixTimestampSeconds().ToString(CultureInfo.InvariantCulture);
        string nonce = Guid.NewGuid().ToString("N").Substring(0, 16);
        string query = BuildCanonicalQuery(parameters);
        string stringToSign = string.Join("\n", new[]
        {
            appKey.Trim(),
            timestamp,
            nonce,
            "GET",
            path,
            query,
            EmptyBodySha256
        });

        string signature;
        try
        {
            signature = ComputeHmacSha256(appSecret.Trim(), stringToSign);
        }
        catch (Exception ex)
        {
            Debug.LogError("机器人接口签名生成失败：" + ex.Message);
            return null;
        }

        string requestUrl = ResolveApiBaseUrl() + path + "?" + query;
        UnityWebRequest request = UnityWebRequest.Get(requestUrl);
        request.SetRequestHeader("X-App-Key", appKey.Trim());
        request.SetRequestHeader("X-Timestamp", timestamp);
        request.SetRequestHeader("X-Nonce", nonce);
        request.SetRequestHeader("X-Sign", signature);
        return request;
    }

    private void HandleSingleStatusResponse(string json)
    {
        RobotOpenStatusResponse response;
        try
        {
            response = JsonUtility.FromJson<RobotOpenStatusResponse>(json);
        }
        catch (Exception ex)
        {
            Debug.LogError("机器人状态 JSON 解析失败：" + ex.Message + "\n" + json);
            return;
        }

        if (!ValidateResponse(response, json) || response.data == null)
        {
            return;
        }

        AcceptApiStatus(response.data, ContainsJsonProperty(json, "angel"));
    }

    private void HandleStatusListResponse(string json)
    {
        RobotOpenStatusListResponse response;
        try
        {
            response = JsonUtility.FromJson<RobotOpenStatusListResponse>(json);
        }
        catch (Exception ex)
        {
            Debug.LogError("机器人列表 JSON 解析失败：" + ex.Message + "\n" + json);
            return;
        }

        if (!ValidateResponse(response, json) || response.data == null || response.data.list == null)
        {
            return;
        }

        RobotOpenStatusData selected = null;
        for (int i = 0; i < response.data.list.Length; i++)
        {
            RobotOpenStatusData item = response.data.list[i];
            if (item == null || string.IsNullOrWhiteSpace(item.robotId))
            {
                continue;
            }

            RegisterDiscoveredRobot(item.robotId, item.robotName, item.houseCode);
            if (selected == null || (!selected.online && item.online))
            {
                selected = item;
            }
        }

        if (selected == null)
        {
            Debug.LogWarning("机器人列表为空。");
            return;
        }

        /*
         * 用户在下拉里手动选了机器人之后，请求列表的那一次可能还在飞行中。
         * 如果回来时无条件写 targetRobotId，就会把用户的选择覆盖回「第一台在线」，
         * 表现就是点完切换、过一秒又跳回原来那台。所以只在还没指定目标时才自动选。
         */
        if (string.IsNullOrWhiteSpace(targetRobotId))
        {
            targetRobotId = selected.robotId;
        }
        else
        {
            RobotOpenStatusData manual = FindRobotById(response.data.list, targetRobotId);
            if (manual == null)
            {
                // 目标不在这份列表里，交给单机状态接口处理，别乱改。
                return;
            }

            selected = manual;
        }

        AcceptApiStatus(selected, ContainsJsonProperty(json, "angel"));
    }

    private static RobotOpenStatusData FindRobotById(RobotOpenStatusData[] list, string robotId)
    {
        if (list == null || string.IsNullOrWhiteSpace(robotId))
        {
            return null;
        }

        string normalized = NormalizeRobotId(robotId);
        for (int i = 0; i < list.Length; i++)
        {
            RobotOpenStatusData item = list[i];
            if (item == null || string.IsNullOrWhiteSpace(item.robotId))
            {
                continue;
            }

            if (string.Equals(
                    NormalizeRobotId(item.robotId),
                    normalized,
                    StringComparison.OrdinalIgnoreCase))
            {
                return item;
            }
        }

        return null;
    }

    private void AcceptApiStatus(RobotOpenStatusData apiData, bool hasHeading)
    {
        RobotStatusMessage status = ConvertStatus(apiData, hasHeading);
        RegisterDiscoveredRobot(status.RobotID, status.Location, status.HouseCode);

        if (!ShouldAcceptRobot(status.RobotID))
        {
            if (logParsedData)
            {
                Debug.Log("忽略其他机器人消息：" + status.RobotID + "，当前目标机器人：" + targetRobotId);
            }
            return;
        }

        // 接口允许不带 B 查询；收到数据后统一使用设备真实编号。
        targetRobotId = status.RobotID;
        HandleRobotStatus(status);
    }

    private static bool ValidateResponse(RobotOpenApiResponseBase response, string json)
    {
        if (response == null)
        {
            Debug.LogWarning("机器人接口 JSON 解析结果为空。\n" + json);
            return false;
        }

        if (response.code != 200)
        {
            Debug.LogWarning("机器人接口返回 code=" + response.code + "，msg=" + response.msg);
            return false;
        }

        return true;
    }

    private bool ShouldAcceptRobot(string robotId)
    {
        if (string.IsNullOrWhiteSpace(targetRobotId))
        {
            return true;
        }

        return string.Equals(
            NormalizeRobotId(targetRobotId),
            NormalizeRobotId(robotId),
            StringComparison.OrdinalIgnoreCase
        );
    }

    private void RegisterDiscoveredRobot(string robotId, string robotName, long houseCode)
    {
        if (string.IsNullOrWhiteSpace(robotId))
        {
            return;
        }

        string normalized = robotId.Trim();
        bool exists = false;

        for (int i = 0; i < discoveredRobotIds.Count; i++)
        {
            if (string.Equals(discoveredRobotIds[i], normalized, StringComparison.OrdinalIgnoreCase))
            {
                exists = true;
                break;
            }
        }

        if (!exists)
        {
            discoveredRobotIds.Add(normalized);
        }

        string label = BuildRobotLabel(robotName, houseCode);
        if (!string.IsNullOrWhiteSpace(label))
        {
            discoveredRobotLabels[normalized] = label;
        }
    }

    /// <summary>
    /// 拼接下拉显示文本。栋舍放在最前面：HUD 那一格只有 225 像素宽，
    /// 超长会被裁掉，而「当前在哪个栋舍」才是挑选机器人时最关键的区分信息。
    /// </summary>
    private static string BuildRobotLabel(string robotName, long houseCode)
    {
        bool hasName = !string.IsNullOrWhiteSpace(robotName);
        bool hasHouse = houseCode > 0L;

        if (hasHouse && hasName)
        {
            return houseCode + "栋 · " + robotName.Trim();
        }

        if (hasHouse)
        {
            return houseCode + "栋";
        }

        if (hasName)
        {
            return robotName.Trim();
        }

        return string.Empty;
    }

    /// <summary>
    /// 机器人 → 栋舍 → 巡检数据。
    ///
    /// 业务链：云端下巡检指令 → 清空「该机器人所在栋舍」的巡检数据 → 场景按栋舍查数据。
    /// 所以切换机器人后，先取状态里的 houseCode，再把它交给 AbnormalChickenApiClient，
    /// 由后者清掉上一个栋舍的残留并按 houseCode 重新拉取。
    /// </summary>
    private void ApplyInspectionHouse(RobotStatusMessage status)
    {
        if (status == null)
        {
            return;
        }

        if (abnormalChickenApiClient == null)
        {
            abnormalChickenApiClient = FindObjectOfType<AbnormalChickenApiClient>();
        }

        /*
         * 单接口模式：后端已经按 robotId 解析出栋舍并把该栋舍的巡检数据一起返回，
         * 这里只把 robotId 传下去。不能再按 houseCode 切一次，
         * 否则两套逻辑会互相清数据、互相改游标。
         */
        if (abnormalChickenApiClient != null &&
            abnormalChickenApiClient.UsesRobotInspectionContext)
        {
            if (!string.IsNullOrWhiteSpace(status.RobotID))
            {
                abnormalChickenApiClient.SwitchRobot(status.RobotID);
            }

            return;
        }

        long house = status.HouseCode;
        if (house <= 0L)
        {
            // 接口没给栋舍时保持现状，不要把已有的过滤条件清掉。
            return;
        }

        if (abnormalChickenApiClient == null)
        {
            abnormalChickenApiClient = FindObjectOfType<AbnormalChickenApiClient>();
        }

        if (abnormalChickenApiClient == null)
        {
            /*
             * 巡检客户端还没创建。
             * 这里不能提前写 currentHouseCode：一旦写了，下一轮 status 会因为
             * house == currentHouseCode 直接短路，之后永远同步不上栋舍。
             */
            return;
        }

        if (house == currentHouseCode)
        {
            return;
        }

        currentHouseCode = house;

        string houseCodeText = house > 0L
            ? house.ToString(CultureInfo.InvariantCulture)
            : "";

        Debug.Log(
            "机器人与栋舍绑定：robotId=" + status.RobotID +
            " → houseCode=" + (string.IsNullOrEmpty(houseCodeText) ? "未提供" : houseCodeText)
        );

        abnormalChickenApiClient.SwitchHouse(houseCodeText);
    }

    private void HandleRobotStatus(RobotStatusMessage status)
    {
        float x = status.Data.x;
        float y = status.Data.y;
        float angel = status.Data.angel;
        float speed = Mathf.Abs(status.Data.Speed);
        int liftPosition = status.Data.LiftPosition;
        bool chargeStatus = status.Data.ChargeStatus;

        RobotRuntimeState newState = AnalyzeRuntimeState(status);

        if (logParsedData)
        {
            Debug.Log(
                "机器人状态：\n" +
                "RobotID = " + status.RobotID + "\n" +
                "Location = " + status.Location + "\n" +
                "x = " + x.ToString("F3") + "\n" +
                "y = " + y.ToString("F3") + "\n" +
                "angel = " + (status.Data.HasHeading ? angel.ToString("F3") : "-- (接口未提供)") + "\n" +
                "speed = " + speed.ToString("F3") + "\n" +
                "LiftPosition = " + liftPosition + "\n" +
                "ChargeStatus = " + chargeStatus + "\n" +
                "CurrentTask = " + status.Data.CurrentTask + "\n" +
                "Inspection = " + status.Data.Inspection + "\n" +
                "RuntimeState = " + newState
            );
        }

        if (logRuntimeState && newState != currentRuntimeState)
        {
            Debug.Log("机器人运行状态变化：" + currentRuntimeState + " → " + newState);
        }

        currentRuntimeState = newState;

        if (poultryTwinHUD == null)
        {
            poultryTwinHUD = FindObjectOfType<PoultryTwinHUDController>();
        }

        if (poultryTwinHUD != null)
        {
            poultryTwinHUD.ApplyRobotStatus(status, currentRuntimeState);
        }

        // 机器人换栋舍后，巡检数据要跟着换：houseCode 变了才重新拉取。
        if (syncInspectionHouseOnRobotSwitch)
        {
            ApplyInspectionHouse(status);
        }

        if (updateUI)
        {
            UpdateRobotUI(status);
        }

        if (status.HasOnlineState && (!status.Online || status.AgeMs > 60000L))
        {
            if (logRuntimeState)
            {
                Debug.LogWarning(
                    "机器人 " + status.RobotID + " 当前离线或数据已过期，保留状态显示但不更新场景位置。" +
                    " ageMs=" + status.AgeMs
                );
            }
            return;
        }

        if (!updateLocatorFromStatusApi)
        {
            hasReceivedFirstStatus = true;
            UpdateLastLift(liftPosition);
            return;
        }

        if (locator == null)
        {
            Debug.LogWarning("RobotOpenApiClient 未绑定 RobotMapLocator。");
            return;
        }

        // 只有响应明确包含 angel 时才应用航向，避免旧服务缺字段时把朝向重置为 0。
        bool applyApiHeading = status.Data.HasHeading &&
                               locator.headingMode == RobotMapLocator.RobotHeadingMode.ApiAngel;

        if (!hasReceivedFirstStatus && firstStatusOnlyAlign)
        {
            locator.SetRobotPoseInstant(x, y, angel, applyApiHeading);
            hasReceivedFirstStatus = true;
            UpdateLastLift(liftPosition);

            if (logRuntimeState)
            {
                Debug.Log("首次状态：已瞬间对齐机器人位置" + (applyApiHeading ? "和朝向。" : "。"));
            }

            return;
        }

        hasReceivedFirstStatus = true;

        if (instantAlignWhenCharging && chargeStatus)
        {
            locator.SetRobotPoseInstant(x, y, angel, applyApiHeading);
            UpdateLastLift(liftPosition);
            return;
        }

        locator.UpdateRobotPose(x, y, angel, speed, applyApiHeading);
        UpdateLastLift(liftPosition);
    }

    private void UpdateRobotUI(RobotStatusMessage status)
    {
        if (status == null || status.Data == null)
        {
            return;
        }

        RobotStatusData data = status.Data;
        float headingDegree = data.angel * Mathf.Rad2Deg;
        float speed = Mathf.Abs(data.Speed);

        if (robotStatusText != null)
        {
            robotStatusText.text =
                "机器人状态\n\n" +
                "设备状态\n" +
                "摄像头开启：" + BoolArrayStatus(data.CameraStatus) + "\n" +
                "继电器开启：" + BoolArrayStatus(data.Relay) + "\n" +
                "巡检状态：" + data.Inspection + "\n\n" +
                "位姿信息\n" +
                "坐标：X=" + data.x.ToString("F2") + "，Y=" + data.y.ToString("F2") + "\n" +
                "航向角：" + (data.HasHeading ? headingDegree.ToString("F2") + "°" : "--") + "\n" +
                "速度：" + speed.ToString("F2") + " m/s\n" +
                "升降台：" + data.LiftPosition + "\n\n" +
                "电源状态\n" +
                "电量：" + data.Battery + "%\n" +
                "充电状态：" + BoolToChinese(data.ChargeStatus);
        }

        if (environmentStatusText != null && data.EnvironmentSensors != null)
        {
            RobotEnvironmentSensors env = data.EnvironmentSensors;

            environmentStatusText.text =
                "环境参数\n" +
                "温度：" + env.Temperate.ToString("F1") + " ℃\n" +
                "湿度：" + env.Humidity.ToString("F1") + " %\n" +
                "CO₂：" + env.CarbonDioxide.ToString("F0") + " ppm\n" +
                "风速：" + env.WindSpeed.ToString("F1") + " m/s\n" +
                "光照：" + env.Light.ToString("F0") + "\n" +
                "氨气：" + env.Ammonia.ToString("F1") + "\n" +
                "PM10：" + env.PM10.ToString("F0") + "\n" +
                "PM2.5：" + env.PM25.ToString("F0");
        }
    }

    private string BoolArrayStatus(bool[] values)
    {
        if (values == null || values.Length == 0)
        {
            return "未知";
        }

        int activeCount = 0;
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i])
            {
                activeCount++;
            }
        }

        return activeCount + "/" + values.Length;
    }

    private string BoolToChinese(bool value)
    {
        return value ? "是" : "否";
    }

    private RobotRuntimeState AnalyzeRuntimeState(RobotStatusMessage status)
    {
        if (status.HasOnlineState && !status.Online)
        {
            return RobotRuntimeState.Offline;
        }

        int lift = status.Data.LiftPosition;

        if (status.Data.ChargeStatus)
        {
            return RobotRuntimeState.Charging;
        }

        if (!hasLastLiftPosition)
        {
            if (lift == 0)
            {
                return RobotRuntimeState.NoPatrolCommand;
            }

            if (lift <= 5)
            {
                return RobotRuntimeState.TaskIssuedLowLift;
            }

            return RobotRuntimeState.InInspectionSegment;
        }

        bool liftRising = lastLiftPosition <= 5 && lift > 5;
        bool liftLowering = lastLiftPosition > 5 && lift <= 5;

        if (lift == 0)
        {
            return RobotRuntimeState.NoPatrolCommand;
        }

        if (liftRising)
        {
            return RobotRuntimeState.LiftingAtInspectionStart;
        }

        if (liftLowering)
        {
            return RobotRuntimeState.LoweringAtInspectionEnd;
        }

        if (lift > 5)
        {
            return RobotRuntimeState.InInspectionSegment;
        }

        return RobotRuntimeState.TaskIssuedLowLift;
    }

    private void UpdateLastLift(int liftPosition)
    {
        lastLiftPosition = liftPosition;
        hasLastLiftPosition = true;
    }

    /// <summary>
    /// 平台限流 120 次/分钟，而且按 appKey 计——巡检接口和状态接口共用这一份配额。
    /// 所以这里给一个下限：状态接口最快 1 秒一次（60 次/分钟），
    /// 留一半给巡检接口，避免两边一起把额度打满后互相拖死（HTTP 429）。
    /// </summary>
    private const float MinimumPollingInterval = 1f;

    /// <summary>
    /// 被限流时放宽轮询间隔。最多放宽到 6 倍（基准 1 秒 → 6 秒），
    /// 只提示一次，避免每轮都刷警告。
    /// </summary>
    private void HandleRateLimited()
    {
        rateLimitBackoff = Mathf.Min(rateLimitBackoff * 2f, 6f);

        if (hasWarnedRateLimit)
        {
            return;
        }

        hasWarnedRateLimit = true;

        Debug.LogWarning(
            "机器人状态接口被平台限流（HTTP 429）：超出 120 次/分钟配额。" +
            "\n已把轮询间隔放宽到 " +
            (Mathf.Max(MinimumPollingInterval, interval) * rateLimitBackoff).ToString("0.0") +
            " 秒。"
        );
    }

    private float GetEffectiveInterval()
    {
        if (interval < MinimumPollingInterval)
        {
            Debug.LogWarning(
                "机器人状态接口轮询间隔 " + interval +
                " 秒过密（平台限流 120 次/分钟），已自动放慢到 " +
                MinimumPollingInterval + " 秒。");
            interval = MinimumPollingInterval;
        }

        return interval * rateLimitBackoff;
    }

    private string ResolveApiBaseUrl()
    {
        string configured = string.IsNullOrWhiteSpace(apiUrl) ? PublicApiBaseUrl : apiUrl.Trim();

        if (configured.IndexOf("/system/brobotreturn/rece", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            string migrated = configured.IndexOf("{host}", StringComparison.OrdinalIgnoreCase) >= 0
                ? LocalApiBaseUrl
                : PublicApiBaseUrl;

            if (!hasWarnedAboutLegacyUrl)
            {
                hasWarnedAboutLegacyUrl = true;
                Debug.LogWarning("检测到旧机器人接口地址，运行时已迁移为 " + migrated);
            }

            return migrated;
        }

        int statusPathIndex = configured.IndexOf(StatusPath, StringComparison.OrdinalIgnoreCase);
        if (statusPathIndex >= 0)
        {
            configured = configured.Substring(0, statusPathIndex);
        }

        return configured.TrimEnd('/');
    }

    private static long GetUnixTimestampSeconds()
    {
        DateTime epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        return (long)(DateTime.UtcNow - epoch).TotalSeconds;
    }

    private static string BuildCanonicalQuery(SortedDictionary<string, string> parameters)
    {
        StringBuilder builder = new StringBuilder();
        foreach (KeyValuePair<string, string> pair in parameters)
        {
            if (builder.Length > 0)
            {
                builder.Append('&');
            }

            builder.Append(Rfc3986Encode(pair.Key));
            builder.Append('=');
            builder.Append(Rfc3986Encode(pair.Value));
        }

        return builder.ToString();
    }

    private static string Rfc3986Encode(string value)
    {
        return Uri.EscapeDataString(value ?? string.Empty).Replace("%7E", "~");
    }

    private static string ComputeHmacSha256(string secret, string value)
    {
        byte[] keyBytes = Encoding.UTF8.GetBytes(secret);
        byte[] valueBytes = Encoding.UTF8.GetBytes(value);

        using (HMACSHA256 hmac = new HMACSHA256(keyBytes))
        {
            byte[] hash = hmac.ComputeHash(valueBytes);
            StringBuilder hex = new StringBuilder(hash.Length * 2);
            for (int i = 0; i < hash.Length; i++)
            {
                hex.Append(hash[i].ToString("x2", CultureInfo.InvariantCulture));
            }

            return hex.ToString();
        }
    }

    private static string NormalizeRobotId(string robotId)
    {
        if (string.IsNullOrWhiteSpace(robotId))
        {
            return string.Empty;
        }

        string normalized = robotId.Trim();
        return normalized.Length > 1 && (normalized[0] == 'B' || normalized[0] == 'b')
            ? normalized.Substring(1)
            : normalized;
    }

    private static bool ContainsJsonProperty(string json, string propertyName)
    {
        return !string.IsNullOrEmpty(json) &&
               json.IndexOf("\"" + propertyName + "\"", StringComparison.Ordinal) >= 0;
    }

    private static RobotStatusMessage ConvertStatus(RobotOpenStatusData source, bool hasHeading)
    {
        RobotEnvironmentSensors environment = null;
        if (source.env != null)
        {
            // 正式文档字段为 pm25；部分已部署服务返回 tsp，兼容两种响应。
            float pm25 = !Mathf.Approximately(source.env.pm25, 0f)
                ? source.env.pm25
                : source.env.tsp;

            environment = new RobotEnvironmentSensors
            {
                Temperate = source.env.temperate,
                Humidity = source.env.humidity,
                Light = source.env.light,
                Ammonia = source.env.ammonia,
                CarbonDioxide = source.env.carbonDioxide,
                WindSpeed = source.env.windSpeed,
                PM10 = source.env.pm10,
                PM25 = pm25,
                // 保留旧字段供尚未迁移的 UI 使用。
                TSP = pm25
            };
        }

        return new RobotStatusMessage
        {
            RobotID = source.robotId,
            Location = source.robotName,
            Message = source.message,
            Online = source.online,
            HasOnlineState = true,
            AgeMs = source.ageMs,
            ReportTime = source.reportTime,
            HouseCode = source.houseCode,
            Data = new RobotStatusData
            {
                x = (float)source.x,
                y = (float)source.y,
                angel = (float)source.angel,
                HasHeading = hasHeading,
                Speed = (float)source.speed,
                LiftPosition = source.liftPosition,
                Battery = source.battery,
                ChargeStatus = source.chargeStatus,
                TaskIndex = source.taskIndex,
                Relay = source.relay,
                CameraStatus = source.cameraStatus,
                Inspection = source.inspection,
                EnvironmentSensors = environment
            }
        };
    }
}

public enum RobotRuntimeState
{
    Unknown,
    Offline,
    NoPatrolCommand,
    TaskIssuedLowLift,
    LiftingAtInspectionStart,
    InInspectionSegment,
    LoweringAtInspectionEnd,
    Charging
}

[Serializable]
public class RobotOpenApiResponseBase
{
    public int code;
    public string msg;
}

[Serializable]
public class RobotOpenStatusResponse : RobotOpenApiResponseBase
{
    public RobotOpenStatusData data;
}

[Serializable]
public class RobotOpenStatusListResponse : RobotOpenApiResponseBase
{
    public RobotOpenStatusListData data;
}

[Serializable]
public class RobotOpenStatusListData
{
    public int total;
    public int onlineCount;
    public int offlineTimeoutSeconds;
    public long serverTime;
    public RobotOpenStatusData[] list;
}

[Serializable]
public class RobotOpenStatusData
{
    public string robotId;
    public string robotName;
    public long houseCode;
    public bool online;
    public long reportTime;
    public string reportTimeText;
    public long ageMs;
    public int battery;
    public int taskIndex;
    public int liftPosition;
    public double x;
    public double y;
    public double angel;
    public double speed;
    public bool chargeStatus;
    public int inspection;
    public bool[] relay;
    public bool[] cameraStatus;
    public RobotOpenEnvironmentData env;
    public string message;
}

[Serializable]
public class RobotOpenEnvironmentData
{
    public float temperate;
    public float humidity;
    public float light;
    public float carbonDioxide;
    public float ammonia;
    public float windSpeed;
    public float pm10;
    public float pm25;
    public float tsp;
}

[Serializable]
public class RobotStatusMessage
{
    public string RobotID;
    public string Location;
    public string Message;
    public bool Online;
    public bool HasOnlineState;
    public long AgeMs;
    public long ReportTime;

    /// <summary>机器人绑定的栋舍编号，来自 /open/robot/status 的 houseCode。</summary>
    public long HouseCode;

    public RobotStatusData Data;
}

[Serializable]
public class RobotStatusData
{
    public float x;
    public float y;
    public float XStart;
    public float YStart;
    public int Channel;

    /// <summary>
    /// 接口字段就是 angel，不是 angle，单位为弧度。
    /// </summary>
    public float angel;
    public bool HasHeading;

    public float Speed;
    public int LiftPosition;
    public int Battery;
    public bool ChargeStatus;
    public bool DriverPort;
    public int TaskCount;
    public int TaskIndex;
    public string CurrentTask;
    public bool[] Relay;
    public bool[] CameraStatus;
    public int Inspection;
    public string ErrorNum;
    public string ErrorMessage;
    public RobotEnvironmentSensors EnvironmentSensors;
}

[Serializable]
public class RobotEnvironmentSensors
{
    /// <summary>
    /// 接口字段是 Temperate，不是 Temperature。
    /// </summary>
    public float Temperate;

    public float Humidity;
    public float Light;
    public float Ammonia;
    public float CarbonDioxide;
    public float WindSpeed;
    public float PM10;
    public float PM25;
    public float TSP;
}
