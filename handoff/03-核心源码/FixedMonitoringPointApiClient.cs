using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// 监测点环境数据客户端（通用版）
///
/// 什么都能接的原因：
///   1. 支持多个数据源（sources 数组），每个源有自己的地址、轮询间隔、超时。
///      以后要接第二家传感器网关，往配置里加一个源就行，不用碰代码。
///   2. 每个源可以配 fieldMap，把「后端字段名」映射到「内部指标名」。
///      换厂家、改字段名都只改配置。
///   3. 没有配 fieldMap 的字段，会自动按别名表匹配四个标准指标；
///      还是不认识的数字字段，自动变成扩展指标（光照、风速……），直接就能显示。
///   4. 返回结构不用约定死：{code,data:[]}、{data:{records:[]}}、{data:{"P01":{}}}、裸数组都认。
///
/// 数据最终统一交给 FixedMonitoringPointSystem.PushSample()，
/// 所以哪天换成 WebSocket 推送，也只要构造 MonitorPointSample 调同一个入口。
/// </summary>
[DisallowMultipleComponent]
[RequireComponent(typeof(FixedMonitoringPointSystem))]
public class FixedMonitoringPointApiClient : MonoBehaviour
{
    public enum DataSourceMode
    {
        /// <summary>走真实后端接口。</summary>
        Backend,
        /// <summary>用本地演示数据（数据卡片会标注「演示」）。</summary>
        Demo
    }

    private const string SettingsResourcePath = "MonitoringPointConfig";
    private const float Missing = float.NaN;

    [Header("数据源")]
    [Tooltip("Backend=走真实接口；Demo=用本地演示数据。")]
    [SerializeField] private DataSourceMode dataSourceMode = DataSourceMode.Backend;

    [Header("兼容旧配置（没有 sources 时使用）")]
    [SerializeField] private string apiUrl = "";
    [SerializeField] private string queryString = "";
    [Min(0.5f)]
    [SerializeField] private float pollingIntervalSeconds = 5f;
    [Min(1)]
    [SerializeField] private int requestTimeoutSeconds = 10;

    [Header("运行时数据源列表")]
    [Tooltip("一般由 MonitoringPointConfig.json 的 sources 填充，也支持 Inspector 手动配。")]
    public List<MonitorDataSourceConfig> sources = new List<MonitorDataSourceConfig>();

    [Header("请求设置")]
    [SerializeField] private bool fetchOnStart = true;
    [SerializeField] private bool enablePolling = true;
    [SerializeField] private bool logRequests = true;
    [Tooltip("接口下发 1/2/3/4 这种编号时，自动映射成 P01/P02。")]
    [SerializeField] private bool normalizeNumericPointId = true;

    private FixedMonitoringPointSystem pointSystem;
    private readonly List<Coroutine> sourceCoroutines = new List<Coroutine>();
    private readonly List<UnityWebRequest> activeRequests = new List<UnityWebRequest>();

    private bool warnedNoSource;
    private int lastSuccessCount;
    private string lastErrorText = "";

    public int LastSuccessCount
    {
        get { return lastSuccessCount; }
    }

    public string LastErrorText
    {
        get { return lastErrorText; }
    }

    /// <summary>是否真的在走后端接口（决定要不要填演示数据）。</summary>
    public bool UsesBackend
    {
        get
        {
            if (dataSourceMode != DataSourceMode.Backend)
            {
                return false;
            }

            List<MonitorDataSourceConfig> resolved = ResolveSources();
            for (int i = 0; i < resolved.Count; i++)
            {
                if (!string.IsNullOrWhiteSpace(resolved[i].apiUrl))
                {
                    return true;
                }
            }

            return false;
        }
    }

    public event Action<int> PointsUpdated;

    #region 生命周期

    private void Awake()
    {
        pointSystem = GetComponent<FixedMonitoringPointSystem>();
        LoadSettingsFromResources();
    }

    private void Start()
    {
        if (dataSourceMode == DataSourceMode.Demo)
        {
            if (logRequests)
            {
                Debug.Log("[监测点接口] 演示数据模式，未请求后端。改 MonitoringPointConfig.json 的 dataSourceMode 为 Backend 即切到真实接口。", this);
            }
            return;
        }

        List<MonitorDataSourceConfig> resolved = ResolveSources();
        if (resolved.Count == 0)
        {
            if (!warnedNoSource)
            {
                warnedNoSource = true;
                Debug.LogWarning(
                    "[监测点接口] 没有可用的数据源。请在 Resources/MonitoringPointConfig.json 的 sources 里配置，" +
                    "或至少填一个 apiUrl。",
                    this
                );
            }
            return;
        }

        if (fetchOnStart)
        {
            RefreshNow();
        }

        if (enablePolling)
        {
            StartPolling();
        }
    }

    private void OnDisable()
    {
        StopPolling();
        AbortActiveRequests();
    }

    #endregion

    #region 配置

    public void LoadSettingsFromResources()
    {
        TextAsset asset = Resources.Load<TextAsset>(SettingsResourcePath);
        if (asset == null || string.IsNullOrWhiteSpace(asset.text))
        {
            return;
        }

        MonitoringPointSettings settings;
        try
        {
            settings = JsonUtility.FromJson<MonitoringPointSettings>(asset.text);
        }
        catch (Exception exception)
        {
            Debug.LogError("[监测点接口] 配置解析失败：" + exception.Message, this);
            return;
        }

        if (settings == null)
        {
            return;
        }

        ApplySettings(settings);

        if (logRequests)
        {
            List<MonitorDataSourceConfig> resolved = ResolveSources();
            Debug.Log(
                "[监测点接口] 已加载配置：mode=" + dataSourceMode +
                "，数据源 " + resolved.Count + " 个" +
                (resolved.Count > 0 ? "（" + resolved[0].name + " → " + resolved[0].apiUrl + "）" : ""),
                this
            );
        }
    }

    public void ApplySettings(MonitoringPointSettings settings)
    {
        if (settings == null)
        {
            return;
        }

        if (settings.sources != null && settings.sources.Count > 0)
        {
            sources = settings.sources;
        }

        if (!string.IsNullOrWhiteSpace(settings.apiUrl)) apiUrl = settings.apiUrl.Trim();
        if (settings.queryString != null) queryString = settings.queryString.Trim();
        if (settings.pollingIntervalSeconds > 0f) pollingIntervalSeconds = settings.pollingIntervalSeconds;
        if (settings.requestTimeoutSeconds > 0) requestTimeoutSeconds = settings.requestTimeoutSeconds;

        fetchOnStart = settings.fetchOnStart;
        enablePolling = settings.enablePolling;
        logRequests = settings.logRequests;
        normalizeNumericPointId = settings.normalizeNumericPointId;

        string mode = (settings.dataSourceMode ?? "").Trim().ToLowerInvariant();
        dataSourceMode = mode == "demo" ? DataSourceMode.Demo : DataSourceMode.Backend;
    }

    /// <summary>
    /// 当前生效的数据源。sources 为空时，用旧的 apiUrl/queryString 拼一个，保证老配置还能跑。
    /// </summary>
    public List<MonitorDataSourceConfig> ResolveSources()
    {
        List<MonitorDataSourceConfig> result = new List<MonitorDataSourceConfig>();

        if (sources != null)
        {
            for (int i = 0; i < sources.Count; i++)
            {
                MonitorDataSourceConfig source = sources[i];
                if (source == null || source.disabled || string.IsNullOrWhiteSpace(source.apiUrl))
                {
                    continue;
                }

                if (source.pollIntervalSeconds <= 0f)
                {
                    source.pollIntervalSeconds = pollingIntervalSeconds;
                }

                if (source.requestTimeoutSeconds <= 0)
                {
                    source.requestTimeoutSeconds = requestTimeoutSeconds;
                }

                if (string.IsNullOrWhiteSpace(source.name))
                {
                    source.name = "source" + i;
                }

                result.Add(source);
            }
        }

        if (result.Count == 0 && !string.IsNullOrWhiteSpace(apiUrl))
        {
            result.Add(new MonitorDataSourceConfig
            {
                name = "default",
                apiUrl = apiUrl,
                queryString = queryString,
                pollIntervalSeconds = pollingIntervalSeconds,
                requestTimeoutSeconds = requestTimeoutSeconds
            });
        }

        return result;
    }

    #endregion

    #region 对外调用

    [ContextMenu("立即请求一次环境数据")]
    public void RefreshNow()
    {
        List<MonitorDataSourceConfig> resolved = ResolveSources();
        if (resolved.Count == 0)
        {
            Debug.LogWarning("[监测点接口] 没有配置数据源，无法请求。", this);
            return;
        }

        for (int i = 0; i < resolved.Count; i++)
        {
            StartCoroutine(RequestSource(resolved[i]));
        }
    }

    public void StartPolling()
    {
        if (sourceCoroutines.Count > 0)
        {
            return;
        }

        List<MonitorDataSourceConfig> resolved = ResolveSources();
        for (int i = 0; i < resolved.Count; i++)
        {
            sourceCoroutines.Add(StartCoroutine(SourceLoop(resolved[i])));
        }
    }

    public void StopPolling()
    {
        for (int i = 0; i < sourceCoroutines.Count; i++)
        {
            if (sourceCoroutines[i] != null)
            {
                StopCoroutine(sourceCoroutines[i]);
            }
        }

        sourceCoroutines.Clear();
    }

    [ContextMenu("切回演示数据")]
    public void SwitchToDemoData()
    {
        StopPolling();
        dataSourceMode = DataSourceMode.Demo;

        if (pointSystem != null)
        {
            pointSystem.FillDemoData();
        }
    }

    #endregion

    #region 轮询

    private IEnumerator SourceLoop(MonitorDataSourceConfig source)
    {
        while (true)
        {
            yield return RequestSource(source);
            yield return new WaitForSecondsRealtime(Mathf.Max(0.5f, source.pollIntervalSeconds));
        }
    }

    private IEnumerator RequestSource(MonitorDataSourceConfig source)
    {
        string url = BuildUrl(source);

        using (UnityWebRequest request = CreateRequest(source, url))
        {
            request.timeout = Mathf.Max(1, source.requestTimeoutSeconds);
            activeRequests.Add(request);

            if (logRequests)
            {
                Debug.Log("[监测点接口][" + source.name + "] " + request.method + " " + url, this);
            }

            yield return request.SendWebRequest();

            activeRequests.Remove(request);

            bool succeeded = request.result == UnityWebRequest.Result.Success;
            string body = request.downloadHandler == null ? "" : request.downloadHandler.text;

            if (!succeeded)
            {
                lastErrorText = "[" + source.name + "] HTTP " + request.responseCode + " " + request.error;
                Debug.LogWarning(
                    "[监测点接口][" + source.name + "] 请求失败：" + lastErrorText +
                    "\n地址：" + url +
                    (string.IsNullOrWhiteSpace(body) ? "" : "\n返回：" + body),
                    this
                );

                yield break;
            }

            if (logRequests)
            {
                Debug.Log("[监测点接口][" + source.name + "] 返回内容：\n" + body, this);
            }

            List<MonitorPointSample> samples;
            string parseError;
            if (!TryParseSamples(body, source, out samples, out parseError))
            {
                lastErrorText = "[" + source.name + "] " + parseError;
                Debug.LogWarning(
                    "[监测点接口][" + source.name + "] 解析失败：" + parseError +
                    "\n原始返回：\n" + body,
                    this
                );

                yield break;
            }

            lastErrorText = "";
            ApplySamples(samples);
        }
    }

    private UnityWebRequest CreateRequest(MonitorDataSourceConfig source, string url)
    {
        string method = (source.method ?? "GET").Trim().ToUpperInvariant();

        if (method == "POST")
        {
            string payload = string.IsNullOrWhiteSpace(source.bodyJson) ? "{}" : source.bodyJson;
            UnityWebRequest request = new UnityWebRequest(url, "POST");
            byte[] bodyRaw = Encoding.UTF8.GetBytes(payload);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            return request;
        }

        return UnityWebRequest.Get(url);
    }

    private string BuildUrl(MonitorDataSourceConfig source)
    {
        string url = source.apiUrl == null ? "" : source.apiUrl.Trim();
        string query = source.queryString == null ? "" : source.queryString.Trim();

        if (query.Length == 0)
        {
            return url;
        }

        if (query[0] == '?')
        {
            query = query.Substring(1);
        }

        return url.IndexOf('?') >= 0 ? url + "&" + query : url + "?" + query;
    }

    private void AbortActiveRequests()
    {
        for (int i = 0; i < activeRequests.Count; i++)
        {
            try
            {
                if (activeRequests[i] != null)
                {
                    activeRequests[i].Abort();
                }
            }
            catch (Exception)
            {
                // Abort 失败不影响后续重试。
            }
        }

        activeRequests.Clear();
    }

    #endregion

    #region 数据落地

    private void ApplySamples(List<MonitorPointSample> samples)
    {
        if (pointSystem == null)
        {
            pointSystem = GetComponent<FixedMonitoringPointSystem>();
        }

        if (pointSystem == null || samples == null)
        {
            return;
        }

        int applied = 0;
        for (int i = 0; i < samples.Count; i++)
        {
            if (pointSystem.PushSample(samples[i]))
            {
                applied++;
            }
        }

        lastSuccessCount = applied;

        if (PointsUpdated != null)
        {
            PointsUpdated(applied);
        }
    }

    #endregion

    #region JSON 解析

    private static readonly string[] PointIdAliases =
    {
        "pointid", "point_id", "pointcode", "point_code", "monitorpointid", "monitor_point_id",
        "deviceid", "device_id", "devid", "dev_id", "devicecode", "device_code",
        "deviceno", "device_no", "sensorid", "sensor_id", "sensorcode", "sensor_code",
        "nodeid", "node_id", "tag", "code", "id", "no", "name"
    };

    private static readonly string[] TemperatureAliases =
    {
        "temperature", "temperaturevalue", "temperature_value", "temp", "tem", "t"
    };

    private static readonly string[] HumidityAliases =
    {
        "humidity", "humidityvalue", "humidity_value", "hum", "rh", "h"
    };

    private static readonly string[] Co2Aliases =
    {
        "co2", "co2value", "co2_value", "co2ppm", "carbondioxide",
        "carbon_dioxide", "co2concentration"
    };

    private static readonly string[] AmmoniaAliases =
    {
        "ammonia", "ammoniavalue", "ammonia_value", "ammoniappm",
        "nh3", "nh3value", "nh3_value", "nh3ppm"
    };

    private static readonly string[] SyncTimeAliases =
    {
        "synctime", "sync_time", "collecttime", "collect_time", "timestamp",
        "updatetime", "update_time", "createtime", "create_time", "time", "ts"
    };

    private static readonly string[] PointNameAliases =
    {
        "pointname", "point_name", "sensorname", "displayname", "display_name"
    };

    private static readonly string[] ContainerAliases =
    {
        "data", "records", "list", "rows", "result", "points", "items", "content"
    };

    /// <summary>
    /// 这些字段是元信息，不参与扩展指标识别，
    /// 否则房间号、状态码、设备名都会被当成指标显示在卡片上。
    /// </summary>
    private static readonly HashSet<string> MetaKeys = BuildMetaKeys();

    private static HashSet<string> BuildMetaKeys()
    {
        HashSet<string> keys = new HashSet<string>();
        keys.UnionWith(PointIdAliases);
        keys.UnionWith(SyncTimeAliases);
        keys.UnionWith(PointNameAliases);

        // 标准指标的别名也算元信息：即使已经通过 fieldMap 拿到值，也不能再漏成扩展指标。
        keys.UnionWith(TemperatureAliases);
        keys.UnionWith(HumidityAliases);
        keys.UnionWith(Co2Aliases);
        keys.UnionWith(AmmoniaAliases);

        keys.Add("housecode");
        keys.Add("house_code");
        keys.Add("tenantid");
        keys.Add("tenant_id");
        keys.Add("status");
        keys.Add("state");
        keys.Add("roomname");
        keys.Add("room_name");
        keys.Add("devicename");
        keys.Add("device_name");
        keys.Add("devicetype");
        keys.Add("device_type");
        keys.Add("msg");
        keys.Add("message");
        keys.Add("code");

        return keys;
    }

    private bool TryParseSamples(
        string json,
        MonitorDataSourceConfig source,
        out List<MonitorPointSample> samples,
        out string error)
    {
        samples = new List<MonitorPointSample>();
        error = "";

        if (string.IsNullOrWhiteSpace(json))
        {
            error = "返回内容为空";
            return false;
        }

        object root;
        try
        {
            root = new JsonReader(json).Parse();
        }
        catch (Exception exception)
        {
            error = "JSON 解析失败：" + exception.Message;
            return false;
        }

        if (root == null)
        {
            error = "返回内容是 null";
            return false;
        }

        IDictionary<string, object> rootDict = root as IDictionary<string, object>;
        if (rootDict != null)
        {
            Dictionary<string, object> lower = LowerKeys(rootDict);
            object codeNode = Pick(lower, "code", "status", "errcode", "errorcode");
            if (codeNode != null)
            {
                float codeValue = ToFloat(codeNode, 200f);
                // RuoYi 成功是 200；也有接口用 0 表示成功，两种都放行。
                if (codeValue != 200f && codeValue != 0f)
                {
                    error = "接口业务返回失败：code=" + ToText(codeNode) +
                            "，msg=" + ToText(Pick(lower, "msg", "message", "error", "errmsg"));
                    return false;
                }
            }
        }

        object container = FindPointContainer(root);
        if (container == null)
        {
            error = "返回内容里找不到监测点数组或监测点对象";
            return false;
        }

        string sourceName = string.IsNullOrWhiteSpace(source.name) ? "backend" : source.name;

        List<object> list = container as List<object>;
        if (list != null)
        {
            for (int i = 0; i < list.Count; i++)
            {
                IDictionary<string, object> item = list[i] as IDictionary<string, object>;
                if (item == null)
                {
                    continue;
                }

                MonitorPointSample sample = ReadSample(item, null, source, sourceName);
                if (sample != null)
                {
                    samples.Add(sample);
                }
            }
        }
        else
        {
            IDictionary<string, object> map = container as IDictionary<string, object>;
            if (map != null)
            {
                // 形态：{ "P01": { "temperature": 24.5 }, "P02": {...} }
                foreach (KeyValuePair<string, object> pair in map)
                {
                    IDictionary<string, object> value = pair.Value as IDictionary<string, object>;
                    if (value == null)
                    {
                        continue;
                    }

                    MonitorPointSample sample = ReadSample(value, pair.Key, source, sourceName);
                    if (sample != null)
                    {
                        samples.Add(sample);
                    }
                }
            }
        }

        if (samples.Count == 0)
        {
            error = "解析成功，但没有任何可用监测点数据";
            return false;
        }

        return true;
    }

    private MonitorPointSample ReadSample(
        IDictionary<string, object> node,
        string fallbackId,
        MonitorDataSourceConfig source,
        string sourceName)
    {
        Dictionary<string, object> lower = LowerKeys(node);
        HashSet<string> consumed = new HashSet<string>();

        // 1. 先按配置的 fieldMap 做显式映射（换厂家只改这里）
        Dictionary<string, float> mapped = new Dictionary<string, float>();
        string mappedId = "";

        if (source.fieldMap != null)
        {
            for (int i = 0; i < source.fieldMap.Count; i++)
            {
                MonitorFieldMapping mapping = source.fieldMap[i];
                if (mapping == null || string.IsNullOrWhiteSpace(mapping.from) || string.IsNullOrWhiteSpace(mapping.to))
                {
                    continue;
                }

                string key = mapping.from.Trim().ToLowerInvariant();
                object value;
                if (!lower.TryGetValue(key, out value) || value == null)
                {
                    continue;
                }

                consumed.Add(key);

                string target = mapping.to.Trim().ToLowerInvariant();

                // to 写 pointId / id 表示「这一列是点位编号」，不是指标。
                if (target == "pointid" || target == "point_id" || target == "id")
                {
                    mappedId = ToText(value);
                    continue;
                }

                float numeric;
                if (TryToFloat(value, out numeric))
                {
                    mapped[target] = numeric;
                }
            }
        }

        // 2. 点位编号：URL 参数兜底 → fieldMap → 内置别名表
        string rawId = fallbackId;

        if (string.IsNullOrWhiteSpace(rawId))
        {
            rawId = mappedId;
        }

        if (string.IsNullOrWhiteSpace(rawId))
        {
            rawId = ToText(PickAndConsume(lower, consumed, PointIdAliases));
        }

        if (string.IsNullOrWhiteSpace(rawId))
        {
            return null;
        }

        string pointId = NormalizePointId(rawId);
        if (string.IsNullOrWhiteSpace(pointId))
        {
            return null;
        }

        MonitorPointSample sample = new MonitorPointSample
        {
            pointId = pointId,
            dataSource = sourceName
        };

        object nameNode = PickAndConsume(lower, consumed, PointNameAliases);
        if (nameNode != null)
        {
            sample.pointName = ToText(nameNode);
        }

        // 3. 写入 fieldMap 映射到的指标
        foreach (KeyValuePair<string, float> pair in mapped)
        {
            sample.SetMetric(pair.Key, pair.Value);
        }

        // 4. fieldMap 没覆盖到的，用内置别名兜底。
        //    注意：即使已经有值也要调一次 PickAndConsume，把这个别名标记成「已消费」，
        //    否则同一个字段会再被当成扩展指标识别一遍。
        object tempNode = PickAndConsume(lower, consumed, TemperatureAliases);
        if (tempNode != null && float.IsNaN(sample.temperature))
        {
            sample.temperature = ToFloat(tempNode, Missing);
        }

        object humNode = PickAndConsume(lower, consumed, HumidityAliases);
        if (humNode != null && float.IsNaN(sample.humidity))
        {
            sample.humidity = ToFloat(humNode, Missing);
        }

        object co2Node = PickAndConsume(lower, consumed, Co2Aliases);
        if (co2Node != null && float.IsNaN(sample.co2))
        {
            sample.co2 = ToFloat(co2Node, Missing);
        }

        object nh3Node = PickAndConsume(lower, consumed, AmmoniaAliases);
        if (nh3Node != null && float.IsNaN(sample.ammonia))
        {
            sample.ammonia = ToFloat(nh3Node, Missing);
        }

        // 5. 采集时间
        object timeNode = PickAndConsume(lower, consumed, SyncTimeAliases);
        sample.syncTime = timeNode == null ? "" : NormalizeSyncTime(ToText(timeNode));

        // 6. 剩下的数字字段自动当扩展指标（光照、风速……加字段不用改代码）
        foreach (KeyValuePair<string, object> pair in lower)
        {
            if (consumed.Contains(pair.Key) || MetaKeys.Contains(pair.Key))
            {
                continue;
            }

            float value;
            if (TryToFloat(pair.Value, out value))
            {
                sample.SetExtraMetric(pair.Key, value);
            }
        }

        // 一条数据都没有的，说明不是环境数据记录，丢掉。
        if (sample.IsEmpty)
        {
            return null;
        }

        return sample;
    }

    #endregion

    #region 工具

    private string NormalizePointId(string raw)
    {
        string text = raw == null ? "" : raw.Trim();
        if (text.Length == 0)
        {
            return "";
        }

        if (!normalizeNumericPointId)
        {
            return text;
        }

        string digits = text;
        if (digits.Length > 1 && (digits[0] == 'P' || digits[0] == 'p'))
        {
            digits = digits.Substring(1);
        }

        int number;
        if (int.TryParse(digits, NumberStyles.Integer, CultureInfo.InvariantCulture, out number))
        {
            if (number > 0 && number < 100)
            {
                return "P" + number.ToString("00", CultureInfo.InvariantCulture);
            }
        }

        return text;
    }

    private static string NormalizeSyncTime(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return "";
        }

        string text = raw.Trim();

        long stamp;
        if (!long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out stamp))
        {
            return text;
        }

        DateTime epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        DateTime moment;

        if (text.Length >= 13)
        {
            moment = epoch.AddMilliseconds(stamp);
        }
        else if (text.Length >= 10)
        {
            moment = epoch.AddSeconds(stamp);
        }
        else
        {
            return text;
        }

        return moment.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);
    }

    private static Dictionary<string, object> LowerKeys(IDictionary<string, object> source)
    {
        Dictionary<string, object> result = new Dictionary<string, object>(source.Count);
        foreach (KeyValuePair<string, object> pair in source)
        {
            if (string.IsNullOrEmpty(pair.Key))
            {
                continue;
            }

            string key = pair.Key.ToLowerInvariant();
            if (!result.ContainsKey(key))
            {
                result[key] = pair.Value;
            }
        }

        return result;
    }

    private static object Pick(IDictionary<string, object> dict, params string[] names)
    {
        if (dict == null || names == null)
        {
            return null;
        }

        for (int i = 0; i < names.Length; i++)
        {
            object value;
            if (dict.TryGetValue(names[i], out value) && value != null)
            {
                return value;
            }
        }

        return null;
    }

    private static object PickAndConsume(
        IDictionary<string, object> dict,
        HashSet<string> consumed,
        params string[] names)
    {
        if (dict == null || names == null)
        {
            return null;
        }

        for (int i = 0; i < names.Length; i++)
        {
            object value;
            if (dict.TryGetValue(names[i], out value) && value != null)
            {
                consumed.Add(names[i]);
                return value;
            }
        }

        return null;
    }

    private static float ToFloat(object node, float fallback)
    {
        float value;
        return TryToFloat(node, out value) ? value : fallback;
    }

    private static bool TryToFloat(object node, out float value)
    {
        value = 0f;

        if (node == null)
        {
            return false;
        }

        if (node is double)
        {
            value = (float)(double)node;
            return true;
        }

        if (node is bool)
        {
            value = ((bool)node) ? 1f : 0f;
            return true;
        }

        string text = node as string;
        if (text != null)
        {
            return float.TryParse(text.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out value);
        }

        return false;
    }

    private static string ToText(object node)
    {
        if (node == null)
        {
            return "";
        }

        string text = node as string;
        if (text != null)
        {
            return text;
        }

        if (node is double)
        {
            double value = (double)node;
            if (Math.Abs(value - Math.Round(value)) < 0.0001)
            {
                return ((long)Math.Round(value)).ToString(CultureInfo.InvariantCulture);
            }

            return value.ToString("0.###", CultureInfo.InvariantCulture);
        }

        if (node is bool)
        {
            return ((bool)node) ? "true" : "false";
        }

        return node.ToString();
    }

    private object FindPointContainer(object node)
    {
        List<object> list = node as List<object>;
        if (list != null)
        {
            return list;
        }

        IDictionary<string, object> dict = node as IDictionary<string, object>;
        if (dict == null)
        {
            return null;
        }

        Dictionary<string, object> lower = LowerKeys(dict);

        for (int i = 0; i < ContainerAliases.Length; i++)
        {
            object child;
            if (!lower.TryGetValue(ContainerAliases[i], out child) || child == null)
            {
                continue;
            }

            if (child is List<object>)
            {
                return child;
            }

            if (child is IDictionary<string, object>)
            {
                object nested = FindPointContainer(child);
                if (nested != null)
                {
                    return nested;
                }
            }
        }

        // 兜底：整条返回本身就是 { "P01": {...} } 这种以点位编号为 key 的结构。
        return dict;
    }

    #endregion

    #region 轻量 JSON 解析器

    /// <summary>
    /// 只做解析、不做序列化的极简 JSON 读取器。
    /// 之所以不用 JsonUtility：它不支持顶层数组，也不支持字段名动态匹配。
    /// </summary>
    private sealed class JsonReader
    {
        private readonly string text;
        private int index;

        public JsonReader(string text)
        {
            this.text = text ?? "";
            index = 0;
        }

        public object Parse()
        {
            SkipWhitespace();
            return ParseValue();
        }

        private object ParseValue()
        {
            SkipWhitespace();

            if (index >= text.Length)
            {
                throw new FormatException("JSON 提前结束");
            }

            char c = text[index];
            switch (c)
            {
                case '{':
                    return ParseObject();
                case '[':
                    return ParseArray();
                case '"':
                    return ParseString();
                case 't':
                    Expect("true");
                    return true;
                case 'f':
                    Expect("false");
                    return false;
                case 'n':
                    Expect("null");
                    return null;
                default:
                    return ParseNumber();
            }
        }

        private Dictionary<string, object> ParseObject()
        {
            Dictionary<string, object> result = new Dictionary<string, object>();

            index++;
            SkipWhitespace();

            if (index < text.Length && text[index] == '}')
            {
                index++;
                return result;
            }

            while (true)
            {
                SkipWhitespace();

                if (index >= text.Length || text[index] != '"')
                {
                    throw new FormatException("JSON 对象缺少 key（位置 " + index + "）");
                }

                string key = ParseString();
                SkipWhitespace();

                if (index >= text.Length || text[index] != ':')
                {
                    throw new FormatException("JSON 对象缺少冒号（位置 " + index + "）");
                }

                index++;
                result[key] = ParseValue();
                SkipWhitespace();

                if (index >= text.Length)
                {
                    throw new FormatException("JSON 对象未闭合");
                }

                if (text[index] == ',')
                {
                    index++;
                    continue;
                }

                if (text[index] == '}')
                {
                    index++;
                    break;
                }

                throw new FormatException("JSON 对象格式错误（位置 " + index + "）");
            }

            return result;
        }

        private List<object> ParseArray()
        {
            List<object> result = new List<object>();

            index++;
            SkipWhitespace();

            if (index < text.Length && text[index] == ']')
            {
                index++;
                return result;
            }

            while (true)
            {
                result.Add(ParseValue());
                SkipWhitespace();

                if (index >= text.Length)
                {
                    throw new FormatException("JSON 数组未闭合");
                }

                if (text[index] == ',')
                {
                    index++;
                    continue;
                }

                if (text[index] == ']')
                {
                    index++;
                    break;
                }

                throw new FormatException("JSON 数组格式错误（位置 " + index + "）");
            }

            return result;
        }

        private string ParseString()
        {
            StringBuilder builder = new StringBuilder();

            index++;

            while (index < text.Length)
            {
                char c = text[index];

                if (c == '"')
                {
                    index++;
                    return builder.ToString();
                }

                if (c != '\\')
                {
                    builder.Append(c);
                    index++;
                    continue;
                }

                index++;
                if (index >= text.Length)
                {
                    break;
                }

                char escape = text[index];
                switch (escape)
                {
                    case 'n':
                        builder.Append('\n');
                        index++;
                        break;
                    case 'r':
                        builder.Append('\r');
                        index++;
                        break;
                    case 't':
                        builder.Append('\t');
                        index++;
                        break;
                    case 'b':
                        builder.Append('\b');
                        index++;
                        break;
                    case 'f':
                        builder.Append('\f');
                        index++;
                        break;
                    case 'u':
                        if (index + 4 < text.Length)
                        {
                            string hex = text.Substring(index + 1, 4);
                            int code;
                            if (int.TryParse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out code))
                            {
                                builder.Append((char)code);
                                index += 5;
                                break;
                            }
                        }

                        builder.Append('u');
                        index++;
                        break;
                    default:
                        builder.Append(escape);
                        index++;
                        break;
                }
            }

            throw new FormatException("JSON 字符串未闭合");
        }

        private object ParseNumber()
        {
            int start = index;

            while (index < text.Length)
            {
                char c = text[index];
                if (char.IsDigit(c) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')
                {
                    index++;
                    continue;
                }

                break;
            }

            string raw = text.Substring(start, index - start);
            if (raw.Length == 0)
            {
                throw new FormatException("JSON 出现非法字符（位置 " + start + "）");
            }

            double value;
            if (!double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
            {
                throw new FormatException("JSON 数字格式错误：" + raw);
            }

            return value;
        }

        private void Expect(string literal)
        {
            if (index + literal.Length > text.Length ||
                string.Compare(text, index, literal, 0, literal.Length, StringComparison.Ordinal) != 0)
            {
                throw new FormatException("JSON 期望 " + literal + "（位置 " + index + "）");
            }

            index += literal.Length;
        }

        private void SkipWhitespace()
        {
            while (index < text.Length && char.IsWhiteSpace(text[index]))
            {
                index++;
            }
        }
    }

    #endregion
}
