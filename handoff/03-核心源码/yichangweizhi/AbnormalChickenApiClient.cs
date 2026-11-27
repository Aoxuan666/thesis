using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

[DisallowMultipleComponent]
[RequireComponent(typeof(AbnormalVisualizer))]
public class AbnormalChickenApiClient : MonoBehaviour
{
    /// <summary>
    /// 与 Postman 测试一致的接口地址。
    /// 默认情况下始终使用该地址，避免 Unity Inspector 保存旧地址。
    /// </summary>
    private const string DefaultApiUrl =
        "http://{host}:{port}/system/CurrentInspectionException";

    [Header("API Address")]

    [Tooltip("关闭时使用代码中的默认接口地址；只有确实需要切换接口时才打开。")]
    [SerializeField]
    private bool useCustomApiUrl = false;

    [Tooltip("自定义接口地址。只有 Use Custom Api Url 打开后才生效。")]
    [SerializeField]
    private string customApiUrl = "";

    /// <summary>
    /// 后端已上线的「按机器人查巡检异常」接口地址。
    /// 直连 8299 最省事：走 nginx 的 /prod-api 前缀不会进签名，容易踩 401。
    /// </summary>
    private const string DefaultRobotInspectionApiUrl =
        "http://{host}:{port}/open/inspection-exception/by-robot";

    [Header("机器人上下文接口")]

    [Tooltip("打开后不再自己拼 houseCode，而是把 robotId 交给后端；" +
             "后端用一个接口完成「robotId → 所在栋舍 → 该栋舍巡检数据」。")]
    [SerializeField]
    private bool useRobotInspectionApi = true;

    [Tooltip("按机器人查巡检异常的接口地址。走 nginx 时也可以填带 /prod-api 的地址，" +
             "签名会自动去掉该前缀。")]
    [SerializeField]
    private string robotInspectionApiUrl = DefaultRobotInspectionApiUrl;

    [Tooltip("当前机器人 ID。切换机器人时由 RobotOpenApiClient 或 HUD 写入。")]
    [SerializeField]
    private string robotId = "";

    [Header("开放接口签名")]

    [Tooltip("开放接口 appKey。单接口模式下必填，缺了服务端直接 401。")]
    [SerializeField]
    private string openApiAppKey = "{APP_KEY}";

    [Tooltip("开放接口 appSecret。仅用于 HMAC-SHA256 签名。" +
             "WebGL 包里这个值是可以被扒出来的，正式对外发布建议改成服务端代理。")]
    [SerializeField]
    private string openApiAppSecret = "{APP_SECRET}";

    [Tooltip("单次条数上限。0 表示不传，由服务端取默认 500。")]
    [SerializeField]
    private int queryLimit = 0;

    [Header("Query Parameters")]

    [Tooltip("租户ID，对应请求参数 tenantId。")]
    [SerializeField]
    private string tenantId = "000000";

    [Tooltip("游标。第一次请求使用0，之后可以自动使用接口返回的lastId。")]
    [SerializeField]
    private long afterId = 0;

    [Tooltip("鸡舍编码。留空表示不传该参数。")]
    [SerializeField]
    private string houseCode = "";

    [Tooltip("异常类型ID。0表示不传该参数，即查询全部异常类型。")]
    [SerializeField]
    private long exceptionTypeId = 0;

    [Header("Request Settings")]

    [Tooltip("进 Play 后自动接入巡检数据流。" +
             "巡检指令由云端下发，孪生端只是数据消费方，不需要用户点按钮才开始。")]
    [SerializeField]
    private bool autoStartOnPlay = true;

    [Tooltip("场景开始运行时立即请求一次。")]
    [SerializeField]
    private bool fetchOnStart = false;

    [Tooltip("场景进入 Play 后自动开始轮询。云端下发巡检指令后要持续收数据，所以默认开着。")]
    [SerializeField]
    private bool enablePolling = true;

    [Tooltip("轮询间隔，单位为秒。默认 3 秒：平台限流 120 次/分钟，且与机器人状态接口共用配额。")]
    [Min(0.1f)]
    [SerializeField]
    private float pollingIntervalSeconds = 3f;

    [Tooltip("接口请求超时时间，单位为秒。")]
    [Min(1)]
    [SerializeField]
    private int requestTimeoutSeconds = 20;

    [Tooltip("请求成功后，根据接口返回结果自动更新afterId。")]
    [SerializeField]
    private bool updateAfterIdAutomatically = true;

    [Header("Debug Settings")]

    [Tooltip("是否在Console中打印完整接口返回内容。" +
             "轮询时会每秒重复一次整包 JSON，只在排查数据问题时打开。")]
    [SerializeField]
    private bool logResponseBody = false;

    [Header("Data Target")]

    [Tooltip("异常鸡可视化组件。一般会自动获取，不需要手动拖拽。")]
    [SerializeField]
    private AbnormalVisualizer abnormalVisualizer;

    private Coroutine requestCoroutine;
    private Coroutine pollingCoroutine;
    private UnityWebRequest activeRequest;
    private readonly HashSet<long> sessionExceptionIds = new HashSet<long>();
    private DateTime inspectionStartedAt;
    private bool waitingForInitialCursor;
    private string lastResolvedHouseCode = "";
    private bool warnedMissingRobotId;

    /*
     * 云端新一轮巡检的判定依据。
     * 平台下发指令时会先清空该栋舍的巡检数据，接口随即返回 total=0；
     * 连续两次空响应 + 本轮确实收到过数据，才认定是「新的一轮开始了」，
     * 避免单次抖动把画面清掉。
     */
    private int emptyResponseStreak;
    private bool receivedRecordsThisRound;
    private bool autoBootstrapDone;

    /*
     * 限流退避：平台限制 120 次/分钟，而且是按 appKey 计（机器人状态接口共用同一份配额）。
     * 平时靠基准间隔控制频率；一旦还是撞上 429，就把间隔按倍数放宽，成功后慢慢收回。
     */
    private float backoffMultiplier = 1f;
    private bool rateLimitLogged;
    private int idleResponseStreak;

    private const float MaxBackoffMultiplier = 8f;

    /// <summary>两次空响应才认定云端已清库，单次抖动不清屏。</summary>
    private const int EmptyResponsesToConfirmNewRound = 2;
    private int sessionDeadCount;
    private int sessionOpenMouthCount;
    private int sessionSoundCount;
    private int sessionStoppedLayingCount;

    /// <summary>服务端返回 truncated=true 时，允许立刻续拉的次数上限，防止死循环。</summary>
    private const int MaxTruncatedFollowUps = 10;

    private bool followUpRequested;
    private int truncatedFollowUps;
    private string lastErrorSignature = "";
    private int repeatedErrorCount;

    /// <summary>
    /// 当前是否正在进行轮询。
    /// </summary>
    public bool IsInspectionRunning
    {
        get { return pollingCoroutine != null; }
    }

    /// <summary>
    /// 当前是否存在HTTP请求。
    /// </summary>
    public bool IsRequesting
    {
        get { return requestCoroutine != null; }
    }

    /// <summary>
    /// 当前使用的afterId。
    /// </summary>
    public long AfterId
    {
        get { return afterId; }
    }

    /// <summary>
    /// 轮询状态变化事件。
    /// </summary>
    public event Action<bool> InspectionRunningChanged;
    public event Action<int, int, int, int> AbnormalCountsChanged;

    #region Unity Lifecycle

    /// <summary>
    /// 进 Play 后自动接入巡检数据流。
    ///
    /// 业务前提：巡检指令由云端下发（云端 → 机器人巡检 → 清空栋舍数据 → 逐条产生异常），
    /// 数字孪生只是消费方，不该等用户点 HUD 上的按钮才开始请求，
    /// 否则一进 Play 就是死屏，只能手动点一下才有数据。
    ///
    /// 这里用 AfterSceneLoad 时机：场景里的 AbnormalVisualizer 和 RobotOpenApiClient
    /// 都已实例化，能直接拿到当前机器人。
    /// </summary>
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void AutoBootstrapInspectionClient()
    {
        /*
         * 编辑器里「打开场景」也会触发 AfterSceneLoad，
         * 那种情况下 AddComponent 会污染场景并标记 dirty，所以只允许 Play 模式接入。
         */
        if (!Application.isPlaying)
        {
            return;
        }

        // 静态方法里显式走 UnityEngine.Object，避免与实例方法的解析混淆。
        AbnormalVisualizer visualizer =
            UnityEngine.Object.FindObjectOfType<AbnormalVisualizer>();

        if (visualizer == null || !visualizer.gameObject.activeInHierarchy)
        {
            return;
        }

        AbnormalChickenApiClient client = visualizer.GetComponent<AbnormalChickenApiClient>();

        if (client == null)
        {
            client = visualizer.gameObject.AddComponent<AbnormalChickenApiClient>();
        }

        if (client == null || client.autoBootstrapDone || !client.autoStartOnPlay)
        {
            return;
        }

        client.autoBootstrapDone = true;
        client.AdoptRobotContextFromScene();

        if (client.enablePolling)
        {
            client.StartInspectionRequests();
        }
        else if (client.fetchOnStart)
        {
            client.FetchAbnormalChickens();
        }
    }

    /// <summary>
    /// 自动接入时，从场景里的机器人客户端取一次当前机器人，
    /// 避免刚进 Play 的头 0.6 秒因为 robotId 为空而空转。
    /// </summary>
    private void AdoptRobotContextFromScene()
    {
        if (!useRobotInspectionApi || !string.IsNullOrWhiteSpace(robotId))
        {
            return;
        }

        RobotOpenApiClient robotClient = FindObjectOfType<RobotOpenApiClient>();

        if (robotClient == null || string.IsNullOrWhiteSpace(robotClient.targetRobotId))
        {
            return;
        }

        robotId = robotClient.targetRobotId.Trim();

        Debug.Log(
            "[AbnormalChickenApiClient] 已自动接入当前机器人：" + robotId,
            this
        );
    }

    private void Reset()
    {
        useCustomApiUrl = false;
        customApiUrl = "";
        useRobotInspectionApi = true;
        robotInspectionApiUrl = DefaultRobotInspectionApiUrl;
        robotId = "";
        openApiAppKey = "{APP_KEY}";
        openApiAppSecret = "{APP_SECRET}";
        queryLimit = 0;
        tenantId = "000000";
        afterId = 0;
        houseCode = "";
        exceptionTypeId = 0;
        autoStartOnPlay = true;
        fetchOnStart = false;
        enablePolling = true;
        pollingIntervalSeconds = 3f;
        requestTimeoutSeconds = 20;
        updateAfterIdAutomatically = true;
        logResponseBody = false;

        ResolveVisualizer();
    }

    private void Awake()
    {
        ResolveVisualizer();
        MigrateLegacyEndpoint();

        if (string.IsNullOrWhiteSpace(tenantId))
        {
            tenantId = "000000";
        }

        if (afterId < 0)
        {
            afterId = 0;
        }
    }

    /// <summary>
    /// 早期版本把这个地址填成了 /open/robot/inspection（当时还是本地 mock 路由），
    /// 真实接口叫 /open/inspection-exception/by-robot。
    /// 场景里如果已经序列化了旧值，运行时会一直 404，这里静默纠正一次。
    /// </summary>
    private void MigrateLegacyEndpoint()
    {
        if (string.IsNullOrWhiteSpace(robotInspectionApiUrl))
        {
            robotInspectionApiUrl = DefaultRobotInspectionApiUrl;
            return;
        }

        string normalized = robotInspectionApiUrl.Trim().TrimEnd('/');

        if (normalized.EndsWith("/open/robot/inspection", StringComparison.OrdinalIgnoreCase))
        {
            Debug.LogWarning(
                "[AbnormalChickenApiClient] 检测到旧的机器人巡检接口地址，已自动切换为 " +
                DefaultRobotInspectionApiUrl,
                this
            );

            robotInspectionApiUrl = DefaultRobotInspectionApiUrl;
        }
    }

    private void Start()
    {
        Debug.Log(
            "[AbnormalChickenApiClient] 当前接口地址：" +
            GetEndpoint() +
            (useRobotInspectionApi
                ? "\n模式：按机器人查询（robotId → 栋舍 → 巡检异常），已启用 HMAC 签名"
                : "\n模式：按租户 + 栋舍查询历史异常（不签名）"),
            this
        );

        /*
         * 组件是 HUD 在点了「开始巡检」时才 AddComponent 出来的，
         * 那种情况下 StartInspectionRequests 已经被 HUD 调过一次，
         * 这里加个判空，避免重复启动刷一条无意义的警告。
         */
        if (enablePolling)
        {
            if (pollingCoroutine == null)
            {
                StartInspectionRequests();
            }
        }
        else if (fetchOnStart)
        {
            FetchAbnormalChickens();
        }
    }

    private void OnDisable()
    {
        StopInspectionRequests();
    }

    private void OnDestroy()
    {
        AbortActiveRequest();
    }

    private void OnValidate()
    {
        if (pollingIntervalSeconds < 0.1f)
        {
            pollingIntervalSeconds = 0.1f;
        }

        if (requestTimeoutSeconds < 1)
        {
            requestTimeoutSeconds = 1;
        }

        if (afterId < 0)
        {
            afterId = 0;
        }
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// 开始异常鸡接口轮询。
    /// 调用后会立即请求一次，然后按照轮询间隔重复请求。
    /// </summary>
    public void StartInspectionRequests()
    {
        if (!Application.isPlaying)
        {
            Debug.LogWarning(
                "[AbnormalChickenApiClient] 请进入Play模式后再启动接口轮询。",
                this
            );
            return;
        }

        if (pollingCoroutine != null)
        {
            /*
             * 自动接入和 HUD 按钮可能都调一次，这里幂等返回，不当成错误。
             */
            return;
        }

        ResolveVisualizer();

        if (abnormalVisualizer == null)
        {
            Debug.LogError(
                "[AbnormalChickenApiClient] 找不到AbnormalVisualizer组件。",
                this
            );
            return;
        }

        if (!useRobotInspectionApi && string.IsNullOrWhiteSpace(tenantId))
        {
            Debug.LogError(
                "[AbnormalChickenApiClient] tenantId不能为空。",
                this
            );
            return;
        }

        enablePolling = true;
        ResetInspectionSession();

        /*
         * 新一轮巡检开始：平台已经清空了该栋舍的巡检数据，
         * 所以这里也必须把游标归零，从头拉本轮全量。
         * 注意不要更早归零——平台清库和轮询之间有时间差，
         * 提前归零可能拉到上一轮的尾巴。
         */
        afterId = 0;
        truncatedFollowUps = 0;
        followUpRequested = false;
        waitingForInitialCursor = true;

        pollingCoroutine = StartCoroutine(PollAbnormalChickens());

        InspectionRunningChanged?.Invoke(true);

        Debug.Log(
            "[AbnormalChickenApiClient] 已开始接收巡检数据（巡检由云端下发，本地只做接收）。" +
            "\n机器人：" + (string.IsNullOrEmpty(robotId) ? "等待状态接口回填" : robotId) +
            "\n轮询间隔：" + pollingIntervalSeconds + "秒",
            this
        );
    }

    /// <summary>
    /// 停止接口轮询和当前正在执行的请求。
    /// </summary>
    public void StopInspectionRequests()
    {
        bool wasRunning = pollingCoroutine != null;

        enablePolling = false;

        if (pollingCoroutine != null)
        {
            StopCoroutine(pollingCoroutine);
            pollingCoroutine = null;
        }

        if (requestCoroutine != null)
        {
            StopCoroutine(requestCoroutine);
            requestCoroutine = null;
        }

        AbortActiveRequest();

        if (wasRunning)
        {
            InspectionRunningChanged?.Invoke(false);

            Debug.Log(
                "[AbnormalChickenApiClient] 已停止接收巡检数据。" +
                "\n注意：只停了孪生端的本地轮询，云端巡检任务本身的启停不在这里控制。",
                this
            );
        }
    }

    /// <summary>
    /// 立即请求一次异常鸡数据。
    /// 可以绑定到Unity按钮。
    /// </summary>
    [ContextMenu("Fetch Abnormal Chickens Now")]
    public void FetchAbnormalChickens()
    {
        if (!Application.isPlaying)
        {
            Debug.LogWarning(
                "[AbnormalChickenApiClient] 请进入Play模式后再请求接口。",
                this
            );
            return;
        }

        ResolveVisualizer();

        if (abnormalVisualizer == null)
        {
            Debug.LogError(
                "[AbnormalChickenApiClient] AbnormalVisualizer没有赋值，" +
                "并且当前场景中也没有找到该组件。",
                this
            );
            return;
        }

        if (!useRobotInspectionApi && string.IsNullOrWhiteSpace(tenantId))
        {
            Debug.LogError(
                "[AbnormalChickenApiClient] tenantId不能为空。",
                this
            );
            return;
        }

        if (useRobotInspectionApi &&
            (string.IsNullOrWhiteSpace(openApiAppKey) || string.IsNullOrWhiteSpace(openApiAppSecret)))
        {
            Debug.LogError(
                "[AbnormalChickenApiClient] 单接口模式需要 appKey / appSecret 才能签名，" +
                "请在 Inspector 的「开放接口签名」里补上。",
                this
            );
            return;
        }

        /*
         * 单接口模式必须知道是哪台机器人，否则后端无法解析栋舍。
         * 启动瞬间 HUD 还没写入 robotId 时会有短暂空档，这里静默等待，
         * 等 SwitchRobot 写进来后自然会重新请求，避免刷一堆 400。
         */
        if (useRobotInspectionApi && string.IsNullOrWhiteSpace(robotId))
        {
            if (!warnedMissingRobotId)
            {
                warnedMissingRobotId = true;
                Debug.LogWarning(
                    "[AbnormalChickenApiClient] 已启用机器人上下文接口，但 robotId 为空；" +
                    "切换一次机器人或手动填 Robot Id 后再请求。",
                    this
                );
            }

            return;
        }

        warnedMissingRobotId = false;

        /*
         * 如果上一次请求还没有结束，
         * 先终止上一次请求，防止多个请求同时执行。
         */
        if (requestCoroutine != null)
        {
            StopCoroutine(requestCoroutine);
            requestCoroutine = null;
        }

        AbortActiveRequest();

        requestCoroutine = StartCoroutine(RequestAbnormalChickens());
    }

    /// <summary>
    /// 接收外部传入的接口JSON。
    /// WebGL JavaScript、WebSocket或者其他组件可以调用该方法。
    /// </summary>
    public bool SubmitResponseJson(string json)
    {
        ResolveVisualizer();

        if (abnormalVisualizer == null)
        {
            Debug.LogError(
                "[AbnormalChickenApiClient] 无法提交JSON，" +
                "因为AbnormalVisualizer不存在。",
                this
            );
            return false;
        }

        if (string.IsNullOrWhiteSpace(json))
        {
            Debug.LogError(
                "[AbnormalChickenApiClient] 接收到的JSON为空。",
                this
            );
            return false;
        }

        try
        {
            return abnormalVisualizer.ReceiveAbnormalChickenResponse(json);
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "[AbnormalChickenApiClient] AbnormalVisualizer处理JSON时发生异常：" +
                "\n" + exception,
                this
            );
            return false;
        }
    }

    /// <summary>
    /// 把afterId恢复为0。
    /// 新一轮巡检开始时可以调用。
    /// </summary>
    [ContextMenu("Reset After ID")]
    public void ResetAfterId()
    {
        afterId = 0;

        Debug.Log(
            "[AbnormalChickenApiClient] afterId已经恢复为0。",
            this
        );
    }

    /// <summary>
    /// 从其他脚本设置afterId。
    /// </summary>
    public void SetAfterId(long value)
    {
        afterId = value < 0 ? 0 : value;
    }

    /// <summary>
    /// 从其他脚本设置租户ID。
    /// </summary>
    public void SetTenantId(string value)
    {
        tenantId = value == null ? "" : value.Trim();
    }

    /// <summary>当前生效的栋舍编号。空字符串表示不按栋舍过滤。</summary>
    public string CurrentHouseCode
    {
        get { return houseCode; }
    }

    /// <summary>
    /// 后端按 robotId 反查出来的栋舍编码（真实生效的查询条件）。
    /// 空字符串表示还没有成功请求过。仅用于显示和排查，不要拿它反过来改查询参数。
    /// </summary>
    public string ResolvedHouseCode
    {
        get { return lastResolvedHouseCode; }
    }

    /// <summary>只改栋舍，不触发请求。</summary>
    public void SetHouseCode(string value)
    {
        houseCode = value == null ? "" : value.Trim();
    }

    /// <summary>
    /// 切换巡检栋舍：机器人换到另一个栋舍后调用。
    ///
    /// 业务上栋舍和机器人是绑定的，云端下发巡检指令时会清空该栋舍的巡检数据，
    /// 所以这里也要把上一个栋舍残留的标记、计数和数据游标一起清掉，
    /// 否则画面里会留着别的栋舍的异常鸡，和接口数据对不上。
    /// </summary>
    public void SwitchHouse(string value)
    {
        string normalized = value == null ? "" : value.Trim();
        bool changed = !string.Equals(houseCode, normalized, StringComparison.Ordinal);
        houseCode = normalized;

        Debug.Log(
            "[AbnormalChickenApiClient] 巡检栋舍：" +
            (string.IsNullOrEmpty(normalized) ? "全部（未按栋舍过滤）" : normalized) +
            (changed ? "，已清空上一个栋舍的数据" : "，未变化"),
            this
        );

        if (!changed)
        {
            return;
        }

        afterId = 0;
        truncatedFollowUps = 0;
        followUpRequested = false;
        lastResolvedHouseCode = "";
        emptyResponseStreak = 0;
        receivedRecordsThisRound = false;
        ClearLocalInspectionData();

        if (Application.isPlaying)
        {
            FetchAbnormalChickens();
        }
    }

    /// <summary>是否启用了「robotId → 栋舍 → 巡检数据」的单接口模式。</summary>
    public bool UsesRobotInspectionContext
    {
        get { return useRobotInspectionApi; }
    }

    /// <summary>当前机器人 ID。空字符串表示未指定。</summary>
    public string CurrentRobotId
    {
        get { return robotId; }
    }

    /// <summary>只改机器人，不触发请求。</summary>
    public void SetRobotId(string value)
    {
        robotId = value == null ? "" : value.Trim();
        warnedMissingRobotId = false;
    }

    /// <summary>
    /// 切换巡检机器人：单接口模式下由 RobotOpenApiClient 或 HUD 调用。
    ///
    /// 与 SwitchHouse 的区别：栋舍不再由客户端解析，只传 robotId，
    /// 但「换机器人要清掉上一台的数据和游标」这件事仍然成立，
    /// 否则上一台机器人所在栋舍的异常鸡会留在画面里。
    /// </summary>
    public void SwitchRobot(string value)
    {
        string normalized = value == null ? "" : value.Trim();
        bool changed = !string.Equals(robotId, normalized, StringComparison.Ordinal);
        robotId = normalized;

        // 状态接口 0.6 秒一次，这里必须只在真正换机器人时动作，否则会反复清数据。
        if (!changed)
        {
            return;
        }

        Debug.Log(
            "[AbnormalChickenApiClient] 巡检机器人：" +
            (string.IsNullOrEmpty(normalized) ? "未指定" : normalized) +
            "，已清空上一台机器人的数据",
            this
        );

        afterId = 0;
        truncatedFollowUps = 0;
        followUpRequested = false;
        lastResolvedHouseCode = "";
        emptyResponseStreak = 0;
        receivedRecordsThisRound = false;
        ClearLocalInspectionData();

        if (Application.isPlaying && !string.IsNullOrEmpty(normalized))
        {
            FetchAbnormalChickens();
        }
    }

    /// <summary>清空场景里的异常标记和统计，但不重置巡检会话的开始时间。</summary>
    private void ClearLocalInspectionData()
    {
        ResolveVisualizer();

        if (abnormalVisualizer != null)
        {
            abnormalVisualizer.ClearMarkers();
        }

        sessionExceptionIds.Clear();
        sessionDeadCount = 0;
        sessionOpenMouthCount = 0;
        sessionSoundCount = 0;
        sessionStoppedLayingCount = 0;
        AbnormalCountsChanged?.Invoke(0, 0, 0, 0);
    }

    /// <summary>
    /// 打印Unity实际生成的请求地址。
    /// </summary>
    [ContextMenu("Print Request URL")]
    public void PrintRequestUrl()
    {
        if (useRobotInspectionApi)
        {
            string signedUrl;
            UnityWebRequest probe = CreateRequest(out signedUrl);

            if (probe != null)
            {
                probe.Dispose();
            }

            Debug.Log(
                "[AbnormalChickenApiClient] 实际请求地址（已签名）：" + signedUrl,
                this
            );
            return;
        }

        Debug.Log(
            "[AbnormalChickenApiClient] 实际请求地址：" +
            BuildRequestUrl(),
            this
        );
    }

    #endregion

    #region Polling

    private IEnumerator PollAbnormalChickens()
    {
        while (isActiveAndEnabled && enablePolling)
        {
            FetchAbnormalChickens();

            /*
             * 等待本次请求结束。
             */
            while (requestCoroutine != null)
            {
                yield return null;
            }

            if (!enablePolling)
            {
                break;
            }

            /*
             * truncated=true 表示这一页被单次上限截断了，
             * 必须立刻用新的 lastId 再拉一次，不能等下一个轮询周期，
             * 否则巡检过程中数据会一段一段地延迟出现。
             */
            if (followUpRequested)
            {
                followUpRequested = false;
                continue;
            }

            yield return new WaitForSecondsRealtime(GetEffectivePollingInterval());
        }

        pollingCoroutine = null;
    }

    /// <summary>
    /// 当前实际生效的轮询间隔。
    ///
    /// 平台限流是 120 次/分钟，且与机器人状态接口共用同一份 appKey 配额：
    ///   巡检 3 秒一次 = 20 次/分钟
    ///   状态 1 秒一次 = 60 次/分钟
    /// 合计 80 次，留出余量给手动刷新和重连。
    /// 万一还是被限流，backoffMultiplier 会把间隔按倍数放宽。
    /// </summary>
    private float GetEffectivePollingInterval()
    {
        return Mathf.Max(0.1f, pollingIntervalSeconds) * backoffMultiplier;
    }

    #endregion

    #region HTTP Request

    private IEnumerator RequestAbnormalChickens()
    {
        string requestUrl;
        UnityWebRequest request = CreateRequest(out requestUrl);

        if (request == null)
        {
            requestCoroutine = null;
            yield break;
        }

        activeRequest = request;
        request.timeout = Mathf.Max(1, requestTimeoutSeconds);

        yield return request.SendWebRequest();

        long responseCode = request.responseCode;

        string responseBody = request.downloadHandler == null
            ? ""
            : request.downloadHandler.text;

        bool requestSucceeded =
            request.result == UnityWebRequest.Result.Success;

        if (!requestSucceeded)
        {
            /*
             * 429 是平台限流（120 次/分钟），不是接口坏了。
             * 单独走退避逻辑：放宽轮询间隔，并且只提示一次，
             * 否则每轮都刷一条红色 Error，控制台会被淹没。
             */
            if (responseCode == 429L)
            {
                HandleRateLimited(responseBody);
            }
            else
            {
                LogRequestError(
                    "HTTP " + responseCode + " / " + request.error,
                    "[AbnormalChickenApiClient] 接口请求失败。" +
                    "\nHTTP状态码：" + responseCode +
                    "\nUnity错误：" + request.error +
                    "\n请求地址：" + requestUrl +
                    "\n服务器返回：" + responseBody +
                    (responseCode == 401L
                        ? "\n排查：签名校验失败。签名 path 不要带 /prod-api，时间戳必须是秒级，nonce 不能复用。"
                        : "")
                );
            }

            FinishRequest(request);
            yield break;
        }

        ResetErrorLog();
        RelaxBackoffAfterSuccess();

        if (logResponseBody)
        {
            Debug.Log(
                "[AbnormalChickenApiClient] 接口返回内容：" +
                "\n" + responseBody,
                this
            );
        }

        if (string.IsNullOrWhiteSpace(responseBody))
        {
            Debug.LogError(
                "[AbnormalChickenApiClient] HTTP请求成功，" +
                "但是接口返回内容为空。",
                this
            );

            FinishRequest(request);
            yield break;
        }

        /*
         * 先检查接口外层结构：
         *
         * {
         *   "code": 200,
         *   "msg": "操作成功",
         *   "data": {
         *      "records": [...]
         *   }
         * }
         */
        ApiResponse apiResponse = null;

        try
        {
            apiResponse = JsonUtility.FromJson<ApiResponse>(responseBody);
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "[AbnormalChickenApiClient] JSON外层结构解析失败：" +
                "\n" + exception +
                "\n原始返回：" + responseBody,
                this
            );
        }

        if (apiResponse == null)
        {
            Debug.LogError(
                "[AbnormalChickenApiClient] 无法解析接口返回结构。",
                this
            );

            FinishRequest(request);
            yield break;
        }

        if (apiResponse.code != 200)
        {
            /*
             * 这个接口的业务错误 HTTP 仍是 200，错误码只写在 body.code 里，
             * 所以必须靠这里判断，不能只看 responseCode。
             */
            LogRequestError(
                "code " + apiResponse.code,
                "[AbnormalChickenApiClient] 接口业务请求失败。" +
                "\n业务状态码：" + apiResponse.code +
                "\n业务消息：" + apiResponse.msg +
                DescribeBusinessError(apiResponse.code, apiResponse.msg) +
                "\n请求地址：" + requestUrl
            );

            FinishRequest(request);
            yield break;
        }

        int recordCount = 0;

        if (apiResponse.data != null &&
            apiResponse.data.records != null)
        {
            recordCount = apiResponse.data.records.Length;
        }

        /*
         * 空轮询会一直重复同一句「记录数 0」，逐条打印会把 Console 刷满。
         * 没数据时只在第一次和每 20 次打一条摘要。
         */
        idleResponseStreak = recordCount > 0 ? 0 : idleResponseStreak + 1;

        if (recordCount > 0 || idleResponseStreak == 1 || idleResponseStreak % 20 == 0)
        {
            Debug.Log(
                "[AbnormalChickenApiClient] 接口业务请求成功。" +
                "\n业务消息：" + apiResponse.msg +
                (useRobotInspectionApi && apiResponse.data != null
                    ? "\n机器人：" + DescribeRobot(apiResponse.data)
                    : "") +
                "\n返回记录数：" + recordCount +
                (recordCount == 0 && idleResponseStreak > 1
                    ? "（已连续 " + idleResponseStreak + " 次为空，日志折叠中）"
                    : "") +
                (useRobotInspectionApi && apiResponse.data != null && apiResponse.data.truncated
                    ? "（已截断，继续续拉）"
                    : ""),
                this
            );
        }

        LogResolvedHouse(apiResponse);

        /*
         * 自动更新afterId。
         */
        if (updateAfterIdAutomatically)
        {
            UpdateAfterId(apiResponse);
        }

        /*
         * truncated=true：本次没取完，标记立刻续拉。
         * 上限保护是为了防止后端 lastId 不前进导致的死循环。
         */
        if (useRobotInspectionApi &&
            apiResponse.data != null &&
            apiResponse.data.truncated)
        {
            if (truncatedFollowUps < MaxTruncatedFollowUps)
            {
                truncatedFollowUps++;
                followUpRequested = true;
            }
            else
            {
                Debug.LogWarning(
                    "[AbnormalChickenApiClient] 连续续拉 " + MaxTruncatedFollowUps +
                    " 次仍是 truncated=true，疑似 lastId 没有前进，先停止续拉等待下一轮轮询。",
                    this
                );
                truncatedFollowUps = 0;
            }
        }
        else
        {
            truncatedFollowUps = 0;
        }

        bool initialCursorResponse = waitingForInitialCursor && IsInspectionRunning;
        if (initialCursorResponse)
        {
            waitingForInitialCursor = false;
        }

        TrackRemoteRoundState(recordCount);

        UpdateSessionCounts(apiResponse.data == null ? null : apiResponse.data.records);

        /*
         * 旧接口（CurrentInspectionException）的语义是「查历史异常」，
         * 首次返回只用来建立本次巡检的游标，不能把历史异常混进本次任务。
         *
         * 新接口（by-robot）的语义是「本轮巡检异常」——平台下发巡检指令时
         * 已经清空该栋舍的缓存表，afterId=0 拉到的就是本轮数据，必须显示，
         * 否则第一屏异常会整批丢掉。
         */
        bool skipFirstScreen = initialCursorResponse && !useRobotInspectionApi;

        bool visualizationSucceeded = skipFirstScreen || recordCount == 0 ||
                                      AppendResponseJson(NormalizeRecordJson(responseBody));

        if (!visualizationSucceeded)
        {
            Debug.LogError(
                "[AbnormalChickenApiClient] 接口返回正常，" +
                "但是AbnormalVisualizer没有成功处理JSON。" +
                "\n请检查ReceiveAbnormalChickenResponse方法中的DTO结构。",
                this
            );
        }

        FinishRequest(request);
    }

    private bool AppendResponseJson(string json)
    {
        ResolveVisualizer();
        if (abnormalVisualizer == null)
        {
            return false;
        }

        try
        {
            return abnormalVisualizer.AppendAbnormalChickenResponse(json);
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "[AbnormalChickenApiClient] 追加异常数据失败：\n" + exception,
                this
            );
            return false;
        }
    }

    private void ResetInspectionSession()
    {
        inspectionStartedAt = DateTime.Now;
        sessionExceptionIds.Clear();
        sessionDeadCount = 0;
        sessionOpenMouthCount = 0;
        sessionSoundCount = 0;
        sessionStoppedLayingCount = 0;

        if (abnormalVisualizer != null)
        {
            abnormalVisualizer.ClearMarkers();
        }

        AbnormalCountsChanged?.Invoke(0, 0, 0, 0);
    }

    /// <summary>
    /// 追踪云端是否已经开启新的一轮巡检。
    ///
    /// 业务链：云端下发指令 → 清空该栋舍的巡检数据 → 机器人逐条产生异常。
    /// 孪生端看不到「指令」本身，只能从数据流反推：
    /// 本轮确实收到过数据，随后连续两次拿到 0 条，说明平台清库了。
    /// 这时必须清掉上一轮的标记和计数，否则画面里会残留上一轮的死鸡。
    /// </summary>
    private void TrackRemoteRoundState(int recordCount)
    {
        if (!useRobotInspectionApi)
        {
            return;
        }

        if (recordCount > 0)
        {
            receivedRecordsThisRound = true;
            emptyResponseStreak = 0;
            return;
        }

        /*
         * 本轮还没收到过任何数据时，空响应不算「新一轮」——
         * 那就是这台机器人所在栋舍本来就干净。
         */
        if (!receivedRecordsThisRound)
        {
            return;
        }

        emptyResponseStreak++;

        if (emptyResponseStreak < EmptyResponsesToConfirmNewRound)
        {
            return;
        }

        HandleRemoteRoundReset();
    }

    private void HandleRemoteRoundReset()
    {
        emptyResponseStreak = 0;
        receivedRecordsThisRound = false;
        truncatedFollowUps = 0;
        followUpRequested = false;
        lastResolvedHouseCode = "";

        /*
         * 游标必须归零：如果平台用 truncate 清表，自增 ID 会从 1 重新开始，
         * 继续用旧的 afterId 会把新数据整批过滤掉。
         */
        afterId = 0;

        ResetInspectionSession();

        Debug.Log(
            "[AbnormalChickenApiClient] 检测到云端已开始新一轮巡检（该栋舍数据被清空）。" +
            "\n已清空上一轮的标记与统计，游标归零重新接收。",
            this
        );
    }

    private void UpdateSessionCounts(ApiRecord[] records)
    {
        if (!IsInspectionRunning || records == null || records.Length == 0)
        {
            return;
        }

        bool changed = false;
        foreach (ApiRecord record in records)
        {
            /*
             * 新接口返回的本来就是「本轮巡检」的数据，不再用本地时钟去卡 syncTime：
             * 服务器和客户端的时钟差会把真实数据误判成上一轮，导致计数一直为 0。
             */
            if (record == null || record.inspectionExceptionId <= 0 ||
                sessionExceptionIds.Contains(record.inspectionExceptionId) ||
                (!useRobotInspectionApi && !BelongsToCurrentInspection(record.syncTime)))
            {
                continue;
            }

            sessionExceptionIds.Add(record.inspectionExceptionId);
            switch (record.exceptionTypeId)
            {
                case 1: sessionDeadCount++; break;
                case 4: sessionSoundCount++; break;
                case 5: sessionOpenMouthCount++; break;
                case 6: sessionStoppedLayingCount++; break;
                default: continue;
            }

            changed = true;
        }

        if (changed)
        {
            AbnormalCountsChanged?.Invoke(
                sessionDeadCount,
                sessionOpenMouthCount,
                sessionSoundCount,
                sessionStoppedLayingCount
            );
        }
    }

    private bool BelongsToCurrentInspection(string syncTime)
    {
        DateTime eventTime;
        if (string.IsNullOrWhiteSpace(syncTime) ||
            !DateTime.TryParse(syncTime, CultureInfo.InvariantCulture, DateTimeStyles.AssumeLocal, out eventTime))
        {
            return true;
        }

        return eventTime >= inspectionStartedAt;
    }

    /// <summary>
    /// 轮询间隔只有 0.6 秒，接口一旦持续报错（比如后端路由还没部署）日志会瞬间刷屏。
    /// 同一类错误只打第一次，之后按 10 / 100 / 1000 次做低频提示。
    /// 请求恢复正常后调用 ResetErrorLog，下次出错还能完整打一次。
    /// </summary>
    private void LogRequestError(string signature, string message)
    {
        if (!string.IsNullOrEmpty(lastErrorSignature) &&
            string.Equals(signature, lastErrorSignature, StringComparison.Ordinal))
        {
            repeatedErrorCount++;

            if (repeatedErrorCount == 10 || repeatedErrorCount % 100 == 0)
            {
                Debug.LogWarning(
                    "[AbnormalChickenApiClient] 同类错误已连续出现 " + repeatedErrorCount +
                    " 次，日志已折叠：\n" + signature,
                    this
                );
            }

            return;
        }

        lastErrorSignature = signature;
        repeatedErrorCount = 0;

        Debug.LogError(message, this);
    }

    private void ResetErrorLog()
    {
        lastErrorSignature = "";
        repeatedErrorCount = 0;
    }

    /// <summary>
    /// 撞上限流时放宽轮询间隔。
    /// 只提示一次（或每次真正放宽一档时提示），避免每轮都刷红色报错。
    /// </summary>
    private void HandleRateLimited(string responseBody)
    {
        float previousInterval = GetEffectivePollingInterval();

        backoffMultiplier = Mathf.Min(backoffMultiplier * 2f, MaxBackoffMultiplier);

        /*
         * 退避已经到顶之后就不再重复提示，否则 3 秒×8=24 秒一次也还是会刷。
         */
        if (rateLimitLogged && Mathf.Approximately(backoffMultiplier, MaxBackoffMultiplier))
        {
            return;
        }

        rateLimitLogged = true;

        Debug.LogWarning(
            "[AbnormalChickenApiClient] 平台限流（HTTP 429）：请求超出 120 次/分钟额度。" +
            "\n服务器返回：" + (string.IsNullOrWhiteSpace(responseBody) ? "(空)" : responseBody) +
            "\n已把轮询间隔从 " + previousInterval.ToString("0.0") +
            " 秒放宽到 " + GetEffectivePollingInterval().ToString("0.0") + " 秒。",
            this
        );
    }

    /// <summary>
    /// 连续成功之后把退避收回一档，直到回到基准间隔。
    /// </summary>
    private void RelaxBackoffAfterSuccess()
    {
        if (backoffMultiplier <= 1f)
        {
            return;
        }

        backoffMultiplier = Mathf.Max(1f, backoffMultiplier * 0.5f);

        if (backoffMultiplier <= 1f && rateLimitLogged)
        {
            rateLimitLogged = false;

            Debug.Log(
                "[AbnormalChickenApiClient] 限流已解除，轮询间隔恢复到 " +
                Mathf.Max(0.1f, pollingIntervalSeconds).ToString("0.0") + " 秒。",
                this
            );
        }
    }

    private void FinishRequest(UnityWebRequest request)
    {
        if (request != null)
        {
            request.Dispose();
        }

        if (activeRequest == request)
        {
            activeRequest = null;
        }

        requestCoroutine = null;
    }

    private void AbortActiveRequest()
    {
        if (activeRequest == null)
        {
            return;
        }

        try
        {
            activeRequest.Abort();
            activeRequest.Dispose();
        }
        catch (Exception exception)
        {
            Debug.LogWarning(
                "[AbnormalChickenApiClient] 终止接口请求时发生异常：" +
                "\n" + exception.Message,
                this
            );
        }
        finally
        {
            activeRequest = null;
        }
    }

    #endregion

    #region URL Building

    /// <summary>
    /// 获取当前使用的接口地址。
    /// </summary>
    private string GetEndpoint()
    {
        if (useRobotInspectionApi &&
            !string.IsNullOrWhiteSpace(robotInspectionApiUrl))
        {
            return robotInspectionApiUrl.Trim();
        }

        if (useCustomApiUrl &&
            !string.IsNullOrWhiteSpace(customApiUrl))
        {
            return customApiUrl.Trim();
        }

        return DefaultApiUrl;
    }

    /// <summary>
    /// 创建请求。
    /// 单接口模式走 HMAC-SHA256 签名；旧接口（CurrentInspectionException）不带签名。
    /// </summary>
    private UnityWebRequest CreateRequest(out string requestUrl)
    {
        requestUrl = string.Empty;

        if (useRobotInspectionApi)
        {
            UnityWebRequest signedRequest;
            string error;

            if (!OpenApiSigner.TryCreateSignedGetRequest(
                    GetEndpoint(),
                    BuildRobotInspectionParameters(),
                    openApiAppKey,
                    openApiAppSecret,
                    out signedRequest,
                    out requestUrl,
                    out error))
            {
                Debug.LogError(
                    "[AbnormalChickenApiClient] 无法创建签名请求：" + error,
                    this
                );

                return null;
            }

            return signedRequest;
        }

        requestUrl = BuildRequestUrl();

        UnityWebRequest request = UnityWebRequest.Get(requestUrl);
        request.SetRequestHeader("Accept", "application/json");
        return request;
    }

    /// <summary>
    /// 单接口模式的参数：只传 robotId / afterId，栋舍由后端按 robotId 反查。
    /// 这几个参数会一起参与签名，所以顺序无所谓，但键值绝不能多也不能少。
    /// </summary>
    private IDictionary<string, string> BuildRobotInspectionParameters()
    {
        SortedDictionary<string, string> parameters =
            new SortedDictionary<string, string>(StringComparer.Ordinal);

        if (!string.IsNullOrWhiteSpace(robotId))
        {
            parameters["robotId"] = robotId.Trim();
        }

        parameters["afterId"] = Math.Max(0L, afterId)
            .ToString(CultureInfo.InvariantCulture);

        if (exceptionTypeId > 0)
        {
            parameters["exceptionTypeId"] = exceptionTypeId
                .ToString(CultureInfo.InvariantCulture);
        }

        if (queryLimit > 0)
        {
            parameters["limit"] = queryLimit
                .ToString(CultureInfo.InvariantCulture);
        }

        return parameters;
    }

    /// <summary>
    /// 生成最终请求地址。
    /// 只用于旧接口（按 tenantId + houseCode 查历史异常）。
    /// </summary>
    private string BuildRequestUrl()
    {
        string endpoint = GetEndpoint();

        string separator;

        if (!endpoint.Contains("?"))
        {
            separator = "?";
        }
        else if (endpoint.EndsWith("?") ||
                 endpoint.EndsWith("&"))
        {
            separator = "";
        }
        else
        {
            separator = "&";
        }

        string safeTenantId =
            UnityWebRequest.EscapeURL(tenantId.Trim());

        string url =
            endpoint +
            separator +
            "tenantId=" +
            safeTenantId;

        url +=
            "&afterId=" +
            Math.Max(0L, afterId)
                .ToString(CultureInfo.InvariantCulture);

        if (useRobotInspectionApi)
        {
            /*
             * 机器人上下文接口：栋舍由后端按 robotId 解析，
             * 客户端不再传 houseCode，避免和后端绑定关系不一致。
             */
            if (!string.IsNullOrWhiteSpace(robotId))
            {
                url +=
                    "&robotId=" +
                    UnityWebRequest.EscapeURL(robotId.Trim());
            }
        }
        else if (!string.IsNullOrWhiteSpace(houseCode))
        {
            url +=
                "&houseCode=" +
                UnityWebRequest.EscapeURL(houseCode.Trim());
        }

        if (exceptionTypeId > 0)
        {
            url +=
                "&exceptionTypeId=" +
                exceptionTypeId.ToString(
                    CultureInfo.InvariantCulture
                );
        }

        return url;
    }

    #endregion

    #region Cursor Handling

    private void UpdateAfterId(ApiResponse response)
    {
        if (response == null || response.data == null)
        {
            return;
        }

        long newAfterId = afterId;

        /*
         * 优先使用后端直接返回的lastId。
         */
        if (response.data.lastId > newAfterId)
        {
            newAfterId = response.data.lastId;
        }

        /*
         * 如果后端没有返回lastId，
         * 就从records中寻找最大的inspectionExceptionId。
         */
        if (response.data.records != null)
        {
            foreach (ApiRecord record in response.data.records)
            {
                if (record == null)
                {
                    continue;
                }

                if (record.inspectionExceptionId > newAfterId)
                {
                    newAfterId = record.inspectionExceptionId;
                }
            }
        }

        if (newAfterId != afterId)
        {
            long oldAfterId = afterId;
            afterId = newAfterId;

            Debug.Log(
                "[AbnormalChickenApiClient] afterId已自动更新：" +
                oldAfterId + " → " + afterId,
                this
            );
        }
    }

    #endregion

    #region Component Resolution

    private void ResolveVisualizer()
    {
        if (abnormalVisualizer != null)
        {
            return;
        }

        abnormalVisualizer = GetComponent<AbnormalVisualizer>();

        if (abnormalVisualizer == null)
        {
            abnormalVisualizer =
                FindObjectOfType<AbnormalVisualizer>();
        }
    }

    #endregion

    #region API DTO

    /// <summary>
    /// 单接口模式下，把后端解析出来的栋舍打印一次（换机器人才打印），
    /// 便于确认「robotId → 栋舍」的绑定是否符合预期。
    /// </summary>
    private void LogResolvedHouse(ApiResponse response)
    {
        if (!useRobotInspectionApi ||
            response == null ||
            response.data == null)
        {
            return;
        }

        /*
         * 真实接口把栋舍直接放在 data.houseCode（字符串，如 "12"）。
         * 兼容一下早期结构 data.house.houseCode（数字），避免老 mock 直接崩。
         */
        string house = response.data.houseCode;

        if (string.IsNullOrWhiteSpace(house) &&
            response.data.house != null &&
            response.data.house.houseCode > 0L)
        {
            house = response.data.house.houseCode.ToString(CultureInfo.InvariantCulture);
        }

        house = house == null ? "" : house.Trim();

        if (house.Length == 0 ||
            string.Equals(house, lastResolvedHouseCode, StringComparison.Ordinal))
        {
            return;
        }

        lastResolvedHouseCode = house;

        Debug.Log(
            "[AbnormalChickenApiClient] 机器人 " +
            (string.IsNullOrEmpty(response.data.robotName)
                ? (string.IsNullOrEmpty(robotId) ? "(未指定)" : robotId)
                : response.data.robotName) +
            " 所在栋舍：" + house +
            "（后续只拉该栋舍的巡检异常）",
            this
        );
    }

    /// <summary>把 data 里的机器人 / 栋舍信息拼成一行，便于日志核对。</summary>
    private static string DescribeRobot(ApiData data)
    {
        if (data == null)
        {
            return "";
        }

        string name = string.IsNullOrWhiteSpace(data.robotName)
            ? (string.IsNullOrWhiteSpace(data.robotIMEI) ? "未命名" : data.robotIMEI)
            : data.robotName;

        string house = string.IsNullOrWhiteSpace(data.houseCode)
            ? "未绑定栋舍"
            : "栋舍 " + data.houseCode;

        return name + " / " + house;
    }

    /// <summary>
    /// 业务错误码（HTTP 仍可能是 200）对应的排查建议。
    /// 这些码是后端文档里明确列出的，直接映射成可执行动作，省得看日志猜。
    /// </summary>
    private static string DescribeBusinessError(int code, string msg)
    {
        switch (code)
        {
            case 400:
                return "\n排查：robotId 不能为空——确认切换机器人后 robotId 已经写进来。";

            case 401:
                return "\n排查：签名校验失败。重点检查：签名 path 是否误带了 /prod-api、" +
                       "时间戳是否秒级、nonce 是否复用、GET 的 body hash 是否为 e3b0c442…。";

            case 404:
                return "\n排查：机器人不存在，或者接口还没部署。" +
                       "\n      返回 \"No endpoint\" 说明服务端没有这个路由；" +
                       "\n      否则先用 /open/robot/status 确认 robotId 有效。";

            case 409:
                return "\n排查：机器人未绑定栋舍（或所属租户缺失），需要在平台侧补录机器人栋舍。";

            case 429:
                return "\n排查：请求过于频繁（默认 120 次/分钟），把轮询间隔调大。";

            case 503:
                return "\n排查：开放接口开关未打开，需要平台侧开启。";

            default:
                return string.IsNullOrWhiteSpace(msg)
                    ? ""
                    : "\n排查：见上方业务消息。";
        }
    }

    /// <summary>
    /// 把 records 里的字段字面量改写成 AbnormalItem 能接受的形式，再交给 AbnormalVisualizer。
    ///
    /// 两件事：
    ///   1. exceptionPosX / exceptionPosY / exceptionJudgmentValue 如果是数字，加引号变字符串
    ///      （AbnormalItem 里它们是 string，JsonUtility 强类型遇 number 会抛 Invalid value）；
    ///   2. exceptionRowNo / ColNo / TierNo 如果是 null，写成 0
    ///      （AbnormalItem 里它们是 int，null 不是合法 int）。
    ///
    /// 只改这几处字面量，不动 JSON 结构，改完仍然合法。
    /// </summary>
    private static string NormalizeRecordJson(string json)
    {
        if (string.IsNullOrEmpty(json) || json.IndexOf('"') < 0)
        {
            return json;
        }

        StringBuilder builder = new StringBuilder(json.Length + 64);
        int index = 0;

        while (index < json.Length)
        {
            char current = json[index];

            if (current != '"')
            {
                builder.Append(current);
                index++;
                continue;
            }

            int stringEnd = FindJsonStringEnd(json, index);

            if (stringEnd < 0)
            {
                builder.Append(json, index, json.Length - index);
                break;
            }

            string token = json.Substring(index, stringEnd - index + 1);
            builder.Append(token);
            index = stringEnd + 1;

            bool toText = IsNumericToStringKey(token);
            bool nullToZero = IsNullToZeroKey(token);

            if (!toText && !nullToZero)
            {
                continue;
            }

            int cursor = index;

            while (cursor < json.Length && char.IsWhiteSpace(json[cursor]))
            {
                builder.Append(json[cursor]);
                cursor++;
            }

            if (cursor >= json.Length || json[cursor] != ':')
            {
                index = cursor;
                continue;
            }

            builder.Append(':');
            cursor++;

            while (cursor < json.Length && char.IsWhiteSpace(json[cursor]))
            {
                builder.Append(json[cursor]);
                cursor++;
            }

            if (cursor >= json.Length)
            {
                index = cursor;
                continue;
            }

            bool isNumber = json[cursor] == '-' ||
                            (json[cursor] >= '0' && json[cursor] <= '9');

            bool isNull = cursor + 4 <= json.Length &&
                          string.CompareOrdinal(json, cursor, "null", 0, 4) == 0;

            if (isNumber)
            {
                int numberEnd = cursor;

                while (numberEnd < json.Length && IsJsonNumberChar(json[numberEnd]))
                {
                    numberEnd++;
                }

                if (toText)
                {
                    builder.Append('"');
                    builder.Append(json, cursor, numberEnd - cursor);
                    builder.Append('"');
                }
                else
                {
                    // 整型字段本来就是数字，原样输出，不能加引号。
                    builder.Append(json, cursor, numberEnd - cursor);
                }

                index = numberEnd;
                continue;
            }

            if (isNull && nullToZero)
            {
                /*
                 * 真实数据里 exceptionColNo 常常是 null（实测 14/14 全是 null），
                 * 而 AbnormalItem 里它是 int。JsonUtility 是强类型的，
                 * 这里直接把 null 写成 0，避免整批解析失败、一条都显示不出来。
                 */
                builder.Append('0');
                index = cursor + 4;
                continue;
            }

            // 本来就是字符串、或非整型字段的 null，原样继续。
            index = cursor;
        }

        return builder.ToString();
    }

    private static bool IsNumericToStringKey(string quotedKey)
    {
        return quotedKey == "\"exceptionPosX\"" ||
               quotedKey == "\"exceptionPosY\"" ||
               quotedKey == "\"exceptionJudgmentValue\"";
    }

    /// <summary>这些是 AbnormalItem 里的整型字段，null 必须换成 0 才能被 JsonUtility 接受。</summary>
    private static bool IsNullToZeroKey(string quotedKey)
    {
        return quotedKey == "\"exceptionRowNo\"" ||
               quotedKey == "\"exceptionColNo\"" ||
               quotedKey == "\"exceptionTierNo\"";
    }

    private static bool IsJsonNumberChar(char c)
    {
        return (c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E';
    }

    /// <summary>找到 JSON 字符串的结束引号，跳过 \" 这类转义。</summary>
    private static int FindJsonStringEnd(string json, int start)
    {
        int index = start + 1;

        while (index < json.Length)
        {
            char c = json[index];

            if (c == '\\')
            {
                index += 2;
                continue;
            }

            if (c == '"')
            {
                return index;
            }

            index++;
        }

        return -1;
    }

    /// <summary>
    /// 只用于检查接口最外层结构和更新afterId。
    /// 完整异常数据仍然交给AbnormalVisualizer解析。
    /// </summary>
    [Serializable]
    private class ApiResponse
    {
        public int code;
        public string msg;
        public ApiData data;
    }

    [Serializable]
    private class ApiData
    {
        public ApiRecord[] records;
        public long lastId;

        // —— 按机器人查巡检异常 /open/inspection-exception/by-robot 的字段 ——

        /// <summary>实际命中的库内 IMEI（可能比你传的多一个 B 前缀）。</summary>
        public string robotIMEI;

        public long robotCode;
        public string robotName;

        /// <summary>后端反查出来的栋舍编码，查询条件就是它。接口里是字符串（"12"），不是数字。</summary>
        public string houseCode;

        public string tenantId;
        public int total;

        /// <summary>true 表示这一页被 limit 截断了，需要立刻拿 lastId 续拉。</summary>
        public bool truncated;

        /// <summary>兼容早期 mock 的嵌套结构，真实接口不返回，留着不影响解析。</summary>
        public ApiHouse house;
    }

    [Serializable]
    private class ApiHouse
    {
        public long houseCode;
        public string houseName;
    }

    [Serializable]
    private class ApiRecord
    {
        public long inspectionExceptionId;
        public int exceptionTypeId;
        public string syncTime;
    }

    #endregion
}
