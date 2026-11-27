using UnityEngine;
using UnityEngine.UI;
using System.Reflection;
using System.Collections.Generic;
using System.Text;

/// <summary>
/// HUD 视角、异常列表与全屏按钮的联动逻辑。
/// </summary>
public class PoultryTwinHUDActions : MonoBehaviour
{
    public WebGLCameraController cameraController;
    public Transform overviewTarget;
    public Transform robotTarget;

    [Tooltip("复位/全景时使用的默认距离。")]
    public float overviewDistance = 30f;
    public float panoramaDistance = 45f;
    public float robotFollowDistance = 10f;
    public float robotFollowYaw = 35f;
    public float robotFollowPitch = 24f;
    public Vector3 robotFollowOffset = new Vector3(0f, 1.2f, 0f);

    private Transform leftPanel;
    private Transform rightPanel;
    private Transform bottomPanel;
    private Transform headerBar;
    private Transform taskBar;
    private Transform viewControls;
    private bool panelsVisible = true;
    private Transform initialOverviewTarget;
    private float initialCameraYaw;
    private float initialCameraPitch;
    private bool hasInitialCameraOrbit;

    private GameObject abnormalListOverlay;
    private Text abnormalListTitle;
    private Text abnormalListBody;
    private GameObject environmentTrendOverlay;
    private GameObject robotControlDrawer;
    private Dropdown robotIdDropdown;

    /*
     * 下拉里显示的是「机器人名 · 栋舍」，真正要切换的是 robotId。
     * Dropdown.OptionData 只有 text，所以用一张平行表按索引取回 id，
     * 不去解析显示文本（文本里已经有人类可读的名称，切不出来）。
     */
    private readonly List<string> robotOptionIds = new List<string>();
    private int lastRobotOptionCount = -1;
    private RobotOpenApiClient cachedRobotApiClient;
    private RobotMotionApiClient robotMotionApiClient;
    private AbnormalChickenApiClient abnormalChickenApiClient;
    private Button automaticModeButton;
    private Button manualModeButton;
    private Button returnChargeButton;
    private Button startInspectionApiButton;
    private Button stopInspectionApiButton;
    private Text inspectionApiStateText;
    private readonly List<Button> manualMotionButtons = new List<Button>();
    private bool manualControlEnabled;

    private readonly Color overlayColor = new Color(0f, 0.08f, 0.12f, 0.42f);
    private readonly Color cardColor = new Color(0.94f, 0.99f, 1f, 0.98f);
    private readonly Color titleColor = new Color(0.06f, 0.19f, 0.26f, 1f);
    private readonly Color bodyColor = new Color(0.14f, 0.27f, 0.32f, 1f);
    private readonly Color accentColor = new Color(0.04f, 0.60f, 0.68f, 1f);
    private readonly Color greenColor = new Color(0.10f, 0.67f, 0.36f, 1f);
    private readonly Color redColor = new Color(0.90f, 0.23f, 0.20f, 1f);

    private void Awake()
    {
        EnsureCrispTextRendering();
        ApplyRobotStatusPanelContrast();
        ResolveSceneReferences();
        CachePanels();
    }

    private void EnsureCrispTextRendering()
    {
        if (GetComponent<CrispUIEnhancer>() == null)
        {
            gameObject.AddComponent<CrispUIEnhancer>();
        }

        Canvas canvas = GetComponentInParent<Canvas>();
        if (canvas != null && canvas.renderMode == RenderMode.ScreenSpaceOverlay)
        {
            canvas.pixelPerfect = true;
        }
    }

    private void ApplyRobotStatusPanelContrast()
    {
        Transform leftPanel = transform.Find("LeftPanel");
        if (leftPanel == null) return;

        Color darkSecondary = new Color(0.08f, 0.20f, 0.24f, 1f);
        Text[] texts = leftPanel.GetComponentsInChildren<Text>(true);
        for (int i = 0; i < texts.Length; i++)
        {
            Text text = texts[i];
            if (text == null) continue;

            string grandParentName = text.transform.parent != null && text.transform.parent.parent != null
                ? text.transform.parent.parent.name
                : string.Empty;

            if (text.name == "Label" &&
                (grandParentName == "DeviceStatusCard" || grandParentName == "PositionCard"))
            {
                text.color = darkSecondary;
            }
            else if (text.name == "ChargeState")
            {
                text.color = darkSecondary;
            }
            else if (text.name == "PreviewText")
            {
                text.color = new Color(0.32f, 0.92f, 0.86f, 1f);
                text.text = text.text.Replace("#9CB5BD", "#D5E8EC");
            }
        }
    }

    private void OnEnable()
    {
        PoultryTwinHUDLanguageSwitcher.LanguageChanged += HandleLanguageChanged;
        ResolveAbnormalChickenApiClient(false);
        BindRobotMotionClient(robotMotionApiClient);
    }

    private void Start()
    {
        /*
         * 巡检数据流由 AbnormalChickenApiClient 在场景加载后自动接入
         * （巡检由云端下发，孪生端不需要点按钮才开始）。
         * 而那时本组件的 OnEnable 已经跑过，事件还没绑上，
         * 所以这里补一次解析，把状态灯和异常计数挂到自动创建的客户端上。
         */
        Invoke(nameof(BindAutoStartedInspectionClient), 0f);
    }

    private void LateUpdate()
    {
        RefreshRobotOptionsIfNeeded();
    }

    private RobotOpenApiClient ResolveRobotApiClient()
    {
        if (cachedRobotApiClient == null)
        {
            cachedRobotApiClient = FindObjectOfType<RobotOpenApiClient>();
        }

        return cachedRobotApiClient;
    }

    /// <summary>
    /// 抽屉打开时，机器人列表可能还在陆续发现。
    /// 原来只在打开的那一瞬间刷一次，先打开抽屉就会一直停在「等待发现机器人…」，
    /// 这里按发现数量变化补刷，不需要关掉再重新打开。
    /// </summary>
    private void RefreshRobotOptionsIfNeeded()
    {
        if (robotIdDropdown == null || robotControlDrawer == null ||
            !robotControlDrawer.activeInHierarchy)
        {
            return;
        }

        RobotOpenApiClient client = ResolveRobotApiClient();
        if (client == null)
        {
            return;
        }

        if (client.DiscoveredRobotIds.Count == lastRobotOptionCount)
        {
            return;
        }

        RefreshRobotOptions();
    }

    private void BindAutoStartedInspectionClient()
    {
        if (abnormalChickenApiClient == null)
        {
            ResolveAbnormalChickenApiClient(false);
        }

        RefreshInspectionApiState();
    }

    private void OnDisable()
    {
        PoultryTwinHUDLanguageSwitcher.LanguageChanged -= HandleLanguageChanged;
        BindInspectionClient(null);
        BindRobotMotionClient(null);
    }

    private void ResolveSceneReferences()
    {
        WebGLCameraController activeController = null;
        if (Camera.main != null && Camera.main.isActiveAndEnabled)
        {
            activeController = Camera.main.GetComponent<WebGLCameraController>();
        }

        if (activeController == null)
        {
            WebGLCameraController[] controllers = FindObjectsOfType<WebGLCameraController>();
            float highestDepth = float.MinValue;
            for (int i = 0; i < controllers.Length; i++)
            {
                Camera controlledCamera = controllers[i].GetComponent<Camera>();
                if (!controllers[i].isActiveAndEnabled || controlledCamera == null || !controlledCamera.isActiveAndEnabled)
                {
                    continue;
                }
                if (controlledCamera.depth >= highestDepth)
                {
                    highestDepth = controlledCamera.depth;
                    activeController = controllers[i];
                }
            }
        }

        if (activeController != null)
        {
            cameraController = activeController;
        }

        if (cameraController != null && initialOverviewTarget == null)
        {
            initialOverviewTarget = cameraController.target;
        }

        if (cameraController != null && !hasInitialCameraOrbit)
        {
            Vector3 euler = cameraController.transform.eulerAngles;
            initialCameraYaw = euler.y;
            initialCameraPitch = euler.x;
            hasInitialCameraOrbit = true;
        }

        if (overviewTarget == null)
        {
            overviewTarget = initialOverviewTarget;
        }

        if (robotTarget == null)
        {
            RobotMapLocator locator = FindObjectOfType<RobotMapLocator>();
            if (locator == null)
            {
                RobotOpenApiClient client = FindObjectOfType<RobotOpenApiClient>();
                if (client != null)
                {
                    locator = client.locator;
                }
            }
            if (locator != null)
            {
                robotTarget = locator.robotRoot;
            }
        }
    }

    /// <summary>Builder 完整生成或局部刷新模块后调用，重新获取各 UI 模块引用。</summary>
    public void CachePanels()
    {
        leftPanel = transform.Find("LeftPanel");
        rightPanel = transform.Find("RightPanel");
        bottomPanel = transform.Find("BottomPanel");
        headerBar = transform.Find("HeaderBar");
        taskBar = transform.Find("TaskStatusBar");
        viewControls = transform.Find("ViewControlBar");
        CacheInspectionControls();
        RefreshInspectionApiState();
    }

    public void StartInspectionApi()
    {
        AbnormalChickenApiClient client = ResolveAbnormalChickenApiClient();
        if (client == null)
        {
            Debug.LogError("HUD: AbnormalChickenApiClient could not be created or found.", this);
            return;
        }

        SyncRobotIdToInspectionClient(client);
        client.StartInspectionRequests();
        RefreshInspectionApiState();
    }

    /// <summary>
    /// 单接口模式靠 robotId 反查栋舍，而巡检客户端是运行时创建的，
    /// 刚创建时 robotId 还是空。这里从机器人开放接口客户端同步一次，
    /// 避免「先点开始巡检、状态轮询还没到」的那几秒请求直接空转。
    /// </summary>
    private void SyncRobotIdToInspectionClient(AbnormalChickenApiClient client)
    {
        if (client == null || !client.UsesRobotInspectionContext)
        {
            return;
        }

        if (!string.IsNullOrWhiteSpace(client.CurrentRobotId))
        {
            return;
        }

        RobotOpenApiClient robotClient = FindObjectOfType<RobotOpenApiClient>();

        if (robotClient == null || string.IsNullOrWhiteSpace(robotClient.targetRobotId))
        {
            return;
        }

        client.SetRobotId(robotClient.targetRobotId.Trim());
    }

    public void StopInspectionApi()
    {
        AbnormalChickenApiClient client = ResolveAbnormalChickenApiClient(false);
        if (client != null)
        {
            client.StopInspectionRequests();
        }

        RefreshInspectionApiState();
    }

    public void ExportAbnormalExcel()
    {
        AbnormalExcelExporter exporter = FindObjectOfType<AbnormalExcelExporter>();
        if (exporter == null)
        {
            AbnormalVisualizer visualizer = FindObjectOfType<AbnormalVisualizer>();
            if (visualizer != null)
            {
                exporter = visualizer.GetComponent<AbnormalExcelExporter>();
                if (exporter == null)
                {
                    exporter = visualizer.gameObject.AddComponent<AbnormalExcelExporter>();
                }
            }
        }

        if (exporter == null)
        {
            Debug.LogError("HUD: AbnormalExcelExporter could not be created or found.", this);
            return;
        }

        exporter.ExportCurrentAbnormalMap();
    }

    private void HandleLanguageChanged(PoultryTwinHUDLanguageSwitcher.Language language)
    {
        RefreshInspectionApiState();
        RefreshReturnChargeButton();
    }

    private void CacheInspectionControls()
    {
        Transform control = bottomPanel != null ? bottomPanel.Find("PatrolControlCard") : null;
        startInspectionApiButton = FindButton(control, "StartInspectionApiButton");
        stopInspectionApiButton = FindButton(control, "StopInspectionApiButton");
        Transform state = control != null ? control.Find("ControlState") : null;
        inspectionApiStateText = state != null ? state.GetComponent<Text>() : null;
    }

    private AbnormalChickenApiClient ResolveAbnormalChickenApiClient(bool createIfMissing = true)
    {
        if (abnormalChickenApiClient == null)
        {
            abnormalChickenApiClient = FindObjectOfType<AbnormalChickenApiClient>();
        }

        if (abnormalChickenApiClient == null && createIfMissing)
        {
            AbnormalVisualizer visualizer = FindObjectOfType<AbnormalVisualizer>();
            if (visualizer != null)
            {
                abnormalChickenApiClient = visualizer.GetComponent<AbnormalChickenApiClient>();
                if (abnormalChickenApiClient == null)
                {
                    abnormalChickenApiClient = visualizer.gameObject.AddComponent<AbnormalChickenApiClient>();
                }

               
            }
        }

        BindInspectionClient(abnormalChickenApiClient);
        return abnormalChickenApiClient;
    }

    private void BindInspectionClient(AbnormalChickenApiClient client)
    {
        if (abnormalChickenApiClient != null)
        {
            abnormalChickenApiClient.InspectionRunningChanged -= HandleInspectionRunningChanged;
            abnormalChickenApiClient.AbnormalCountsChanged -= HandleAbnormalCountsChanged;
        }

        abnormalChickenApiClient = client;

        if (abnormalChickenApiClient != null)
        {
            abnormalChickenApiClient.InspectionRunningChanged += HandleInspectionRunningChanged;
            abnormalChickenApiClient.AbnormalCountsChanged += HandleAbnormalCountsChanged;
        }
    }

    private void HandleInspectionRunningChanged(bool running)
    {
        RefreshInspectionApiState();
    }

    private void HandleAbnormalCountsChanged(int dead, int openMouth, int sound, int stoppedLaying)
    {
        PoultryTwinHUDController hud = GetComponent<PoultryTwinHUDController>();
        if (hud != null)
        {
            hud.SetAbnormalCounts(dead, openMouth, sound, stoppedLaying);
        }
    }

    private void RefreshInspectionApiState()
    {
        AbnormalChickenApiClient client = ResolveAbnormalChickenApiClient(false);
        bool running = client != null && client.IsInspectionRunning;
        bool english = PoultryTwinHUDLanguageSwitcher.IsEnglish();

        if (inspectionApiStateText != null)
        {
            inspectionApiStateText.text = english
                ? (running ? "\u25cf API polling" : "\u25cf API stopped")
                : (running ? "\u25cf \u63a5\u53e3\u8bf7\u6c42\u4e2d" : "\u25cf \u672a\u8bf7\u6c42\u63a5\u53e3");
            inspectionApiStateText.color = running ? greenColor : bodyColor;
        }

        if (startInspectionApiButton != null)
        {
            startInspectionApiButton.interactable = !running;
        }

        if (stopInspectionApiButton != null)
        {
            stopInspectionApiButton.interactable = running;
        }
    }

    private static Button FindButton(Transform parent, string objectName)
    {
        Transform child = parent != null ? parent.Find(objectName) : null;
        return child != null ? child.GetComponent<Button>() : null;
    }

    public void ResetView()
    {
        ResolveSceneReferences();
        if (cameraController == null)
        {
            Debug.LogWarning("HUD：未指定 WebGLCameraController，无法复位视角。");
            return;
        }

        cameraController.followTarget = false;
        cameraController.distance = overviewDistance;

        if (overviewTarget != null)
        {
            SetCameraOrbit(overviewTarget, overviewDistance, initialCameraYaw, initialCameraPitch, Vector3.zero, false);
        }

        Debug.Log("HUD：复位视角，目标=" + (overviewTarget != null ? overviewTarget.name : "当前观察中心") + "，距离=" + overviewDistance);
    }

    public void FollowRobot()
    {
        ResolveSceneReferences();
        if (cameraController == null || robotTarget == null)
        {
            Debug.LogWarning("HUD：未指定 Camera Controller 或 RobotRoot，无法跟随机器人。");
            return;
        }

        SetCameraOrbit(robotTarget, robotFollowDistance, robotFollowYaw, robotFollowPitch, robotFollowOffset, true);
        Debug.Log("HUD：开始跟随机器人，目标=" + robotTarget.name);
    }

    public void Overview()
    {
        ResolveSceneReferences();
        if (cameraController == null)
        {
            Debug.LogWarning("HUD：未找到 WebGLCameraController，无法进入全景模式。");
            return;
        }

        cameraController.followTarget = false;
        cameraController.distance = panoramaDistance;
        Transform target = overviewTarget != null ? overviewTarget : initialOverviewTarget;
        if (target != null)
        {
            SetCameraOrbit(target, panoramaDistance, initialCameraYaw, initialCameraPitch, Vector3.zero, false);
        }
        Debug.Log("HUD：进入全景模式，目标=" + (target != null ? target.name : "当前观察中心") + "，距离=" + panoramaDistance);
    }

    private void SetCameraOrbit(Transform target, float distance, float yaw, float pitch, Vector3 offset, bool follow)
    {
        cameraController.target = target;
        cameraController.distance = Mathf.Clamp(distance, cameraController.minDistance, cameraController.maxDistance);
        cameraController.followOffset = offset;
        cameraController.followTarget = follow;

        BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        typeof(WebGLCameraController).GetField("yaw", flags)?.SetValue(cameraController, yaw);
        typeof(WebGLCameraController).GetField("pitch", flags)?.SetValue(cameraController, pitch);
        typeof(WebGLCameraController).GetField("targetPosition", flags)?.SetValue(cameraController, target.position + offset);
        cameraController.FocusOnTarget(target);
    }

    public void ToggleSidePanels()
    {
        panelsVisible = !panelsVisible;
        SetActive(leftPanel, panelsVisible);
        SetActive(rightPanel, panelsVisible);
        SetActive(bottomPanel, panelsVisible);
        SetActive(headerBar, panelsVisible);
        SetActive(taskBar, panelsVisible);

        // 视角控制条始终保留，方便再次打开 UI。
        SetActive(viewControls, true);
    }

    /// <summary>顶部“告警”按钮：直接打开当前任务告警列表。</summary>
    public void ShowAlarmList()
    {
        ShowAbnormalOverlay("告警事件（本次任务）");
    }

    /// <summary>顶部“异常列表”按钮：打开或关闭本次任务异常列表。</summary>
    public void ToggleAbnormalList()
    {
        EnsureAbnormalOverlay();
        if (abnormalListOverlay.activeSelf)
        {
            abnormalListOverlay.SetActive(false);
            return;
        }

        ShowAbnormalOverlay("异常列表（本次任务）");
    }

    /// <summary>顶部“全屏”按钮：构建为 WebGL 后由浏览器进入/退出全屏。</summary>
    public void ToggleFullscreen()
    {
        Screen.fullScreen = !Screen.fullScreen;
        Debug.Log("HUD：已请求切换全屏。Unity Editor 的 Game 视图通常不会出现明显变化；请在 WebGL 页面中测试。");
    }

    public void CloseAbnormalList()
    {
        if (abnormalListOverlay != null)
        {
            abnormalListOverlay.SetActive(false);
        }
    }

    public void LocateFirstAbnormal()
    {
        AbnormalMarker[] markers = FindObjectsOfType<AbnormalMarker>();
        if (markers == null || markers.Length == 0)
        {
            Debug.LogWarning("当前场景中没有可定位的异常标记。");
            return;
        }

        ResolveSceneReferences();
        if (cameraController == null)
        {
            Debug.LogWarning("未找到场景相机控制器，无法定位异常。");
            return;
        }

        cameraController.followTarget = false;
        cameraController.distance = Mathf.Clamp(8f, cameraController.minDistance, cameraController.maxDistance);
        cameraController.FocusOnTarget(markers[0].transform);
    }

    public void ToggleEnvironmentTrend()
    {
        EnsureEnvironmentTrendOverlay();
        environmentTrendOverlay.SetActive(!environmentTrendOverlay.activeSelf);

        if (environmentTrendOverlay.activeSelf)
        {
            PoultryTwinHUDController hud = GetComponent<PoultryTwinHUDController>();
            if (hud != null)
            {
                hud.RefreshVisuals();
            }
        }
    }

    public void CloseEnvironmentTrend()
    {
        if (environmentTrendOverlay != null)
        {
            environmentTrendOverlay.SetActive(false);
        }
    }

    public void ToggleRobotControl()
    {
        EnsureRobotControlDrawer();
        RefreshRobotOptions();
        robotControlDrawer.SetActive(!robotControlDrawer.activeSelf);
        robotControlDrawer.transform.SetAsLastSibling();
        SyncCurrentLanguage();
    }

    public void CloseRobotControl()
    {
        if (robotControlDrawer != null) robotControlDrawer.SetActive(false);
    }

    private void EnsureRobotControlDrawer()
    {
        if (robotControlDrawer != null) return;

        Font font = ResolveFont();
        robotControlDrawer = CreateUI("RobotControlDrawer", transform, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        RectTransform drawerRt = robotControlDrawer.GetComponent<RectTransform>();
        Pin(drawerRt, new Vector2(1f, 0.5f), new Vector2(1f, 0.5f), new Vector2(1f, 0.5f), new Vector2(-18f, -12f), new Vector2(350f, 650f));
        Image drawerImage = robotControlDrawer.GetComponent<Image>();
        drawerImage.color = new Color(0.94f, 0.99f, 1f, 0.98f);
        drawerImage.raycastTarget = true;
        Outline outline = robotControlDrawer.AddComponent<Outline>();
        outline.effectColor = new Color(0.18f, 0.65f, 0.72f, 0.85f);
        outline.effectDistance = new Vector2(1.2f, -1.2f);

        Text title = CreateText("Title", robotControlDrawer.transform, "机器人控制", font, 20, titleColor, TextAnchor.MiddleLeft, FontStyle.Bold);
        Pin(title.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0.5f, 1f), new Vector2(0f, -25f), new Vector2(-86f, 32f));
        Button close = CreateButton("CloseButton", robotControlDrawer.transform, "×", new Color(0.78f, 0.87f, 0.89f, 1f), font);
        Pin(close.GetComponent<RectTransform>(), new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-14f, -14f), new Vector2(34f, 28f));
        close.onClick.AddListener(CloseRobotControl);

        RobotOpenApiClient apiClient = FindObjectOfType<RobotOpenApiClient>();
        robotIdDropdown = CreateMetricDropdown(robotControlDrawer.transform, font);
        robotIdDropdown.gameObject.name = "RobotIdDropdown";
        Pin(robotIdDropdown.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(16f, -63f), new Vector2(225f, 30f));
        robotIdDropdown.options.Clear();
        robotIdDropdown.options.Add(new Dropdown.OptionData(apiClient != null && !string.IsNullOrWhiteSpace(apiClient.targetRobotId) ? apiClient.targetRobotId : "等待发现机器人..."));
        robotIdDropdown.RefreshShownValue();
        Button switchRobot = CreateButton("SwitchRobotButton", robotControlDrawer.transform, "切换", accentColor, font);
        Pin(switchRobot.GetComponent<RectTransform>(), new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-16f, -63f), new Vector2(80f, 30f));
        switchRobot.onClick.AddListener(ApplyRobotSelection);

        GameObject modeCard = CreateUI("ModeCard", robotControlDrawer.transform, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        Pin(modeCard.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0.5f, 1f), new Vector2(0f, -98f), new Vector2(-28f, 88f));
        modeCard.GetComponent<Image>().color = new Color(0.90f, 0.96f, 0.97f, 0.72f);

        GameObject moveCard = CreateUI("MoveCard", robotControlDrawer.transform, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        Pin(moveCard.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0.5f, 1f), new Vector2(0f, -192f), new Vector2(-28f, 190f));
        moveCard.GetComponent<Image>().color = new Color(0.90f, 0.96f, 0.97f, 0.58f);

        GameObject taskCard = CreateUI("TaskCard", robotControlDrawer.transform, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        Pin(taskCard.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0.5f, 1f), new Vector2(0f, -388f), new Vector2(-28f, 126f));
        taskCard.GetComponent<Image>().color = new Color(0.90f, 0.96f, 0.97f, 0.72f);

        Text modeTitle = CreateText("ModeTitle", robotControlDrawer.transform, "控制模式", font, 15, titleColor, TextAnchor.MiddleLeft, FontStyle.Bold);
        Pin(modeTitle.rectTransform, new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(16f, -108f), new Vector2(100f, 22f));
        Button automatic = CreateCommandButton(robotControlDrawer.transform, "✓ 自动巡检", accentColor, font, new Vector2(86f, -153f), "自动巡检");
        Button manual = CreateCommandButton(robotControlDrawer.transform, "手动控制", new Color(0.28f, 0.43f, 0.49f, 1f), font, new Vector2(238f, -153f), "手动控制");
        automaticModeButton = automatic;
        manualModeButton = manual;

        Text moveTitle = CreateText("MoveTitle", robotControlDrawer.transform, "移动控制", font, 15, titleColor, TextAnchor.MiddleLeft, FontStyle.Bold);
        Pin(moveTitle.rectTransform, new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(16f, -200f), new Vector2(120f, 22f));
        CreateRoundCommand(robotControlDrawer.transform, "↑", font, new Vector2(175f, -243f), "前进");
        CreateRoundCommand(robotControlDrawer.transform, "←", font, new Vector2(119f, -299f), "左移");
        CreateRoundCommand(robotControlDrawer.transform, "■", font, new Vector2(175f, -299f), "停止");
        CreateRoundCommand(robotControlDrawer.transform, "→", font, new Vector2(231f, -299f), "右移");
        CreateRoundCommand(robotControlDrawer.transform, "↓", font, new Vector2(175f, -355f), "后退");

        Text rotateTitle = CreateText("RotateTitle", robotControlDrawer.transform, "转向与任务", font, 15, titleColor, TextAnchor.MiddleLeft, FontStyle.Bold);
        Pin(rotateTitle.rectTransform, new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(16f, -400f), new Vector2(140f, 22f));
        CreateCommandButton(robotControlDrawer.transform, "↶ 左转", accentColor, font, new Vector2(86f, -448f), "左转");
        CreateCommandButton(robotControlDrawer.transform, "右转 ↷", accentColor, font, new Vector2(238f, -448f), "右转");
        returnChargeButton = CreateCommandButton(robotControlDrawer.transform, "返回充电", new Color(0.28f, 0.43f, 0.49f, 1f), font, new Vector2(86f, -494f), "返回充电");
        CreateCommandButton(robotControlDrawer.transform, "紧急停止", redColor, font, new Vector2(238f, -494f), "紧急停止");

        Text hint = CreateText("Hint", robotControlDrawer.transform, "手动控制启用后：按住移动，松开停止", font, 11, new Color(bodyColor.r, bodyColor.g, bodyColor.b, 0.70f), TextAnchor.MiddleCenter, FontStyle.Normal);
        Pin(hint.rectTransform, new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(0.5f, 0f), new Vector2(0f, 15f), new Vector2(-28f, 28f));
        SetManualControl(false);
        BindRobotMotionClient(ResolveRobotMotionApiClient());
        RefreshReturnChargeButton();
        robotControlDrawer.SetActive(false);
        SyncCurrentLanguage();
    }

    private void ApplyRobotSelection()
    {
        RobotOpenApiClient client = ResolveRobotApiClient();
        if (client == null || robotIdDropdown == null)
        {
            Debug.LogWarning("HUD：未找到 RobotOpenApiClient，无法切换机器人。");
            return;
        }

        int index = robotIdDropdown.value;
        if (index < 0 || index >= robotOptionIds.Count)
        {
            Debug.LogWarning("HUD：尚未从接口消息中发现机器人 ID，请稍后再试。");
            return;
        }

        string selected = robotOptionIds[index];
        if (string.IsNullOrWhiteSpace(selected))
        {
            Debug.LogWarning("HUD：尚未从接口消息中发现机器人 ID，请稍后再试。");
            return;
        }

        selected = selected.Trim();
        client.targetRobotId = selected;
        client.ResetRuntimeState();

        // 单接口模式：不等下一次状态回来，立刻按新机器人重新拉该栋舍的巡检数据。
        AbnormalChickenApiClient abnormalClient = FindObjectOfType<AbnormalChickenApiClient>();
        if (abnormalClient != null && abnormalClient.UsesRobotInspectionContext)
        {
            abnormalClient.SwitchRobot(selected);
        }

        PoultryTwinHUDController hud = GetComponent<PoultryTwinHUDController>();
        if (hud != null) hud.ResetForRobotSelection();
        Debug.Log(
            "HUD：已切换目标机器人为 " + client.GetDiscoveredRobotLabel(selected) +
            "（" + selected + "）");
    }

    private void RefreshRobotOptions()
    {
        RobotOpenApiClient client = ResolveRobotApiClient();
        if (client == null || robotIdDropdown == null)
        {
            return;
        }

        string current = client.targetRobotId;
        robotIdDropdown.options.Clear();
        robotOptionIds.Clear();

        // 先按「栋舍序号 → 名称」排序，几十台机器人时肉眼才找得到要的那台。
        List<KeyValuePair<string, string>> entries = new List<KeyValuePair<string, string>>();
        for (int i = 0; i < client.DiscoveredRobotIds.Count; i++)
        {
            string id = client.DiscoveredRobotIds[i];
            string label = client.GetDiscoveredRobotLabel(id);
            entries.Add(new KeyValuePair<string, string>(
                id,
                string.IsNullOrWhiteSpace(label) ? id : label));
        }

        entries.Sort((a, b) => CompareRobotOptionLabel(a.Value, b.Value));

        for (int i = 0; i < entries.Count; i++)
        {
            robotOptionIds.Add(entries[i].Key);
            robotIdDropdown.options.Add(new Dropdown.OptionData(entries[i].Value));
        }

        lastRobotOptionCount = robotOptionIds.Count;

        if (robotOptionIds.Count == 0)
        {
            robotIdDropdown.options.Add(new Dropdown.OptionData("等待发现机器人..."));
            robotIdDropdown.SetValueWithoutNotify(0);
            robotIdDropdown.RefreshShownValue();
            return;
        }

        int selectedIndex = 0;
        for (int i = 0; i < robotOptionIds.Count; i++)
        {
            if (string.Equals(robotOptionIds[i], current, System.StringComparison.OrdinalIgnoreCase))
            {
                selectedIndex = i;
                break;
            }
        }
        robotIdDropdown.SetValueWithoutNotify(selectedIndex);
        robotIdDropdown.RefreshShownValue();
    }

    /// <summary>标签形如「49栋 · 12舍巡检机器人」，先比栋舍数字，再比文字。</summary>
    private static int CompareRobotOptionLabel(string left, string right)
    {
        int leftHouse = ReadLeadingNumber(left);
        int rightHouse = ReadLeadingNumber(right);

        if (leftHouse >= 0 && rightHouse >= 0 && leftHouse != rightHouse)
        {
            return leftHouse.CompareTo(rightHouse);
        }

        return string.CompareOrdinal(left, right);
    }

    private static int ReadLeadingNumber(string text)
    {
        if (string.IsNullOrEmpty(text) || !char.IsDigit(text[0]))
        {
            return -1;
        }

        int value = 0;
        int i = 0;
        while (i < text.Length && i < 6 && char.IsDigit(text[i]))
        {
            value = value * 10 + (text[i] - '0');
            i++;
        }

        return value;
    }

    private Button CreateCommandButton(Transform parent, string label, Color color, Font font, Vector2 position, string command)
    {
        Button button = CreateButton(command + "Button", parent, label, color, font);
        Pin(button.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0.5f, 0.5f), position, new Vector2(134f, 34f));
        if (!BindHoldMotion(button, command))
        {
            button.onClick.AddListener(() => ExecuteRobotMotionCommand(command));
        }
        return button;
    }

    private void CreateRoundCommand(Transform parent, string label, Font font, Vector2 position, string command)
    {
        Button button = CreateButton(command + "Button", parent, label, command == "停止" ? redColor : accentColor, font);
        Pin(button.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0.5f, 0.5f), position, new Vector2(46f, 46f));
        if (!BindHoldMotion(button, command))
        {
            button.onClick.AddListener(() => ExecuteRobotMotionCommand(command));
        }
    }

    private bool BindHoldMotion(Button button, string command)
    {
        int direction;
        switch (command)
        {
            case "前进": direction = 0; break;
            case "后退": direction = 1; break;
            case "右移":
            case "右转": direction = 2; break;
            case "左移":
            case "左转": direction = 3; break;
            default: return false;
        }

        RobotMotionHoldButton holdButton = button.gameObject.AddComponent<RobotMotionHoldButton>();
        holdButton.Configure(ResolveRobotMotionApiClient(), direction, () => manualControlEnabled);
        manualMotionButtons.Add(button);
        return true;
    }

    /// <summary>
    /// 供外部脚本（Web 桥接层、外部自动化脚本）触发机器人命令。
    ///
    /// 走的是和 HUD 按钮完全相同的一条路径，所以「自动巡检模式下不能手动控制」
    /// 这类约束、按钮置灰、控制通道切换都会一并生效，不会出现 Web 端绕开限制乱发指令。
    ///
    /// 支持的命令："前进" / "后退" / "左转" / "右转" / "返回充电" / "自动巡检" / "手动控制"
    /// </summary>
    public void ApplyRobotCommand(string command)
    {
        if (string.IsNullOrEmpty(command))
        {
            return;
        }

        ExecuteRobotMotionCommand(command.Trim());
    }

    /// <summary>当前是否处于手动控制模式。外部脚本可据此决定要不要显示控制按钮。</summary>
    public bool IsManualControlEnabled
    {
        get { return manualControlEnabled; }
    }

    private void ExecuteRobotMotionCommand(string command)
    {
        RobotMotionApiClient client = ResolveRobotMotionApiClient();
        switch (command)
        {
            case "自动巡检":
                SetManualControl(false);
                return;
            case "手动控制":
                SetManualControl(true);
                return;
            case "返回充电":
                client.ReturnToChargingDock();
                return;
        }

        if (!manualControlEnabled)
        {
            Debug.LogWarning("机器人当前处于自动巡检模式，请先切换到手动控制。");
            return;
        }

        switch (command)
        {
            case "前进":
                client.MoveForward();
                break;
            case "后退":
                client.MoveBackward();
                break;
            case "左移":
            case "左转":
                client.TurnLeft();
                break;
            case "右移":
            case "右转":
                client.TurnRight();
                break;
            default:
                Debug.Log("机器人控制命令：" + command + "（接口尚未实际验证，未发送）");
                break;
        }
    }

    private void SetManualControl(bool enabled)
    {
        manualControlEnabled = enabled;
        Color inactiveColor = new Color(0.28f, 0.43f, 0.49f, 1f);

        ConfigureRobotDataSource(enabled);

        SetModeButtonVisual(automaticModeButton, !enabled, "自动巡检", inactiveColor);
        SetModeButtonVisual(manualModeButton, enabled, "手动控制", inactiveColor);

        for (int i = 0; i < manualMotionButtons.Count; i++)
        {
            if (manualMotionButtons[i] != null)
            {
                manualMotionButtons[i].interactable = enabled;
            }
        }

        SyncCurrentLanguage();
    }

    private void SyncCurrentLanguage()
    {
        PoultryTwinHUDLanguageSwitcher languageSwitcher = GetComponent<PoultryTwinHUDLanguageSwitcher>();
        if (languageSwitcher == null)
        {
            return;
        }

        languageSwitcher.Initialize();
        languageSwitcher.ApplyLanguage(PoultryTwinHUDLanguageSwitcher.CurrentLanguage);
    }

    private void ConfigureRobotDataSource(bool useSlamtecDirectControl)
    {
        RobotOpenApiClient statusClient = FindObjectOfType<RobotOpenApiClient>();
        if (statusClient == null) return;

        // 手动控制如果走后端转发，就不需要为了拿位姿去连底盘 WiFi。
        RobotMotionApiClient motionClient = FindObjectOfType<RobotMotionApiClient>();
        if (useSlamtecDirectControl &&
            motionClient != null &&
            motionClient.IsUsingBackendChannel)
        {
            useSlamtecDirectControl = false;
            Debug.Log("手动控制走后端转发，定位继续使用服务端上报，无需连接底盘 WiFi。");
        }

        statusClient.useRobotLocalPose = useSlamtecDirectControl;
        statusClient.updateLocatorFromStatusApi = !useSlamtecDirectControl;

        RobotLocalPoseClient localPoseClient = statusClient.GetComponent<RobotLocalPoseClient>();
        if (useSlamtecDirectControl)
        {
            if (localPoseClient == null)
            {
                localPoseClient = statusClient.gameObject.AddComponent<RobotLocalPoseClient>();
            }
            localPoseClient.Configure(statusClient.locator);
            localPoseClient.enabled = true;
        }
        else if (localPoseClient != null)
        {
            localPoseClient.enabled = false;
        }

        statusClient.ResetRuntimeState();
        Debug.Log(useSlamtecDirectControl
            ? "机器人控制模式：手动控制，运动与定位使用思岚接口。"
            : "机器人控制模式：自动巡检，定位使用原服务端接口。");
    }

    private void SetModeButtonVisual(Button button, bool selected, string label, Color inactiveColor)
    {
        if (button == null) return;
        Image image = button.GetComponent<Image>();
        if (image != null) image.color = selected ? accentColor : inactiveColor;
        Text text = button.GetComponentInChildren<Text>();
        if (text != null) text.text = selected ? "✓ " + label : label;
    }

    private RobotMotionApiClient ResolveRobotMotionApiClient()
    {
        RobotMotionApiClient resolved = robotMotionApiClient;
        if (resolved == null)
        {
            resolved = FindObjectOfType<RobotMotionApiClient>();
        }
        if (resolved == null)
        {
            resolved = gameObject.AddComponent<RobotMotionApiClient>();
        }

        BindRobotMotionClient(resolved);
        return robotMotionApiClient;
    }

    private void BindRobotMotionClient(RobotMotionApiClient client)
    {
        if (robotMotionApiClient != null)
        {
            robotMotionApiClient.GoHomeStateChanged -= HandleGoHomeStateChanged;
        }

        robotMotionApiClient = client;

        if (robotMotionApiClient != null && isActiveAndEnabled)
        {
            robotMotionApiClient.GoHomeStateChanged += HandleGoHomeStateChanged;
        }
    }

    private void HandleGoHomeStateChanged(RobotMotionApiClient.GoHomeState state, string message)
    {
        RefreshReturnChargeButton();
    }

    private void RefreshReturnChargeButton()
    {
        if (returnChargeButton == null)
        {
            return;
        }

        RobotMotionApiClient.GoHomeState state = robotMotionApiClient != null
            ? robotMotionApiClient.CurrentGoHomeState
            : RobotMotionApiClient.GoHomeState.Idle;
        bool active = state == RobotMotionApiClient.GoHomeState.Creating ||
                      state == RobotMotionApiClient.GoHomeState.Returning;
        bool english = PoultryTwinHUDLanguageSwitcher.IsEnglish();
        Text label = returnChargeButton.GetComponentInChildren<Text>();

        if (label != null)
        {
            switch (state)
            {
                case RobotMotionApiClient.GoHomeState.Creating:
                    label.text = english ? "Submitting..." : "提交中...";
                    break;
                case RobotMotionApiClient.GoHomeState.Returning:
                    label.text = english ? "Returning..." : "回充中...";
                    break;
                case RobotMotionApiClient.GoHomeState.Succeeded:
                    label.text = english ? "At Dock" : "已到充电桩";
                    break;
                case RobotMotionApiClient.GoHomeState.Failed:
                    label.text = english ? "Retry Return" : "重试回充";
                    break;
                default:
                    label.text = english ? "Return" : "返回充电";
                    break;
            }
        }

        returnChargeButton.interactable = !active;
        Image image = returnChargeButton.GetComponent<Image>();
        if (image != null)
        {
            image.color = state == RobotMotionApiClient.GoHomeState.Succeeded
                ? greenColor
                : (state == RobotMotionApiClient.GoHomeState.Failed
                    ? redColor
                    : (active ? accentColor : new Color(0.28f, 0.43f, 0.49f, 1f)));
        }
    }

    private void EnsureEnvironmentTrendOverlay()
    {
        if (environmentTrendOverlay != null)
        {
            return;
        }

        Font font = ResolveFont();
        environmentTrendOverlay = CreateUI("EnvironmentTrendOverlay", transform, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        RectTransform overlayRt = environmentTrendOverlay.GetComponent<RectTransform>();
        Stretch(overlayRt, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
        Image overlayImage = environmentTrendOverlay.GetComponent<Image>();
        overlayImage.color = overlayColor;
        overlayImage.raycastTarget = true;

        GameObject card = CreateUI("TrendCard", environmentTrendOverlay.transform, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        RectTransform cardRt = card.GetComponent<RectTransform>();
        Pin(cardRt, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), Vector2.zero, new Vector2(820f, 480f));
        card.GetComponent<Image>().color = cardColor;
        Outline outline = card.AddComponent<Outline>();
        outline.effectColor = new Color(0.32f, 0.76f, 0.81f, 0.90f);
        outline.effectDistance = new Vector2(1.2f, -1.2f);

        Text title = CreateText("Title", card.transform, "环境变化趋势（最近30分钟）", font, 24, titleColor, TextAnchor.MiddleLeft, FontStyle.Bold);
        Pin(title.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0.5f, 1f), new Vector2(0f, -30f), new Vector2(-54f, 36f));

        Dropdown metricDropdown = CreateMetricDropdown(card.transform, font);
        Pin(metricDropdown.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(30f, -58f), new Vector2(170f, 32f));

        GameObject chartObject = CreateUI("ExpandedEnvironmentTrendChart", card.transform, typeof(RectTransform), typeof(CanvasRenderer), typeof(EnvironmentTrendChart));
        Stretch(chartObject.GetComponent<RectTransform>(), Vector2.zero, Vector2.one, new Vector2(110f, 54f), new Vector2(-44f, -92f));
        PoultryTwinHUDController hud = GetComponent<PoultryTwinHUDController>();
        if (hud != null)
        {
            hud.expandedEnvironmentTrendChart = chartObject.GetComponent<EnvironmentTrendChart>();
            hud.expandedEnvironmentTrendChart.SetDisplayMode(EnvironmentTrendChart.DisplayMode.Temperature);
            metricDropdown.onValueChanged.AddListener(hud.SetExpandedTrendMetric);
            hud.expandedEnvironmentTrendTimeText = null;
            hud.expandedEnvironmentTimeLabels = new Text[6];
            for (int i = 0; i < hud.expandedEnvironmentTimeLabels.Length; i++)
            {
                float anchor = i / (float)(hud.expandedEnvironmentTimeLabels.Length - 1);
                TextAnchor alignment = i == 0 ? TextAnchor.MiddleLeft : (i == hud.expandedEnvironmentTimeLabels.Length - 1 ? TextAnchor.MiddleRight : TextAnchor.MiddleCenter);
                Text timeLabel = CreateText("TimeTick" + i, card.transform, "--:--:--", font, 13, bodyColor, alignment, FontStyle.Normal);
                Pin(timeLabel.rectTransform, new Vector2(anchor, 0f), new Vector2(anchor, 0f), new Vector2(anchor, 0f), new Vector2(Mathf.Lerp(30f, -30f, anchor), 33f), new Vector2(92f, 24f));
                hud.expandedEnvironmentTimeLabels[i] = timeLabel;
            }

            const int axisLabelCount = 4;
            hud.expandedTemperatureAxisLabels = new Text[axisLabelCount];
            hud.expandedHumidityAxisLabels = new Text[axisLabelCount];
            for (int i = 0; i < axisLabelCount; i++)
            {
                float anchor = i / (float)(axisLabelCount - 1);
                Text tempLabel = CreateText("TemperatureTick" + i, card.transform, "--", font, 12, new Color(0.91f, 0.29f, 0.22f, 1f), TextAnchor.MiddleRight, FontStyle.Bold);
                Pin(tempLabel.rectTransform, new Vector2(0f, anchor), new Vector2(0f, anchor), new Vector2(0f, anchor), new Vector2(58f, Mathf.Lerp(57f, -96f, anchor)), new Vector2(48f, 20f));
                hud.expandedTemperatureAxisLabels[i] = tempLabel;

                Text humidityLabel = CreateText("HumidityTick" + i, card.transform, "--", font, 12, new Color(0.10f, 0.46f, 0.88f, 1f), TextAnchor.MiddleRight, FontStyle.Bold);
                Pin(humidityLabel.rectTransform, new Vector2(0f, anchor), new Vector2(0f, anchor), new Vector2(0f, anchor), new Vector2(58f, Mathf.Lerp(57f, -96f, anchor)), new Vector2(48f, 20f));
                hud.expandedHumidityAxisLabels[i] = humidityLabel;
            }
        }

        Button close = CreateButton("CloseButton", card.transform, "关闭", redColor, font);
        Pin(close.GetComponent<RectTransform>(), new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-22f, -20f), new Vector2(88f, 34f));
        close.onClick.AddListener(CloseEnvironmentTrend);
        environmentTrendOverlay.SetActive(false);
    }

    private Dropdown CreateMetricDropdown(Transform parent, Font font)
    {
        GameObject root = CreateUI("MetricDropdown", parent, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image), typeof(Dropdown));
        Image background = root.GetComponent<Image>();
        background.color = new Color(0.90f, 0.97f, 0.98f, 1f);
        Dropdown dropdown = root.GetComponent<Dropdown>();

        Text caption = CreateText("Label", root.transform, "温度（℃）", font, 14, titleColor, TextAnchor.MiddleLeft, FontStyle.Bold);
        Stretch(caption.rectTransform, Vector2.zero, Vector2.one, new Vector2(12f, 2f), new Vector2(-30f, -2f));
        Text arrow = CreateText("Arrow", root.transform, "▼", font, 12, accentColor, TextAnchor.MiddleCenter, FontStyle.Bold);
        Pin(arrow.rectTransform, new Vector2(1f, 0.5f), new Vector2(1f, 0.5f), new Vector2(1f, 0.5f), new Vector2(-8f, 0f), new Vector2(24f, 24f));

        GameObject template = CreateUI("Template", root.transform, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image), typeof(ScrollRect));
        RectTransform templateRt = template.GetComponent<RectTransform>();
        Pin(templateRt, new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(0.5f, 1f), new Vector2(0f, -2f), new Vector2(0f, 132f));
        template.GetComponent<Image>().color = Color.white;
        GameObject viewport = CreateUI("Viewport", template.transform, typeof(RectTransform), typeof(RectMask2D));
        Stretch(viewport.GetComponent<RectTransform>(), Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
        GameObject content = CreateUI("Content", viewport.transform, typeof(RectTransform));
        RectTransform contentRt = content.GetComponent<RectTransform>();
        Pin(contentRt, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0.5f, 1f), Vector2.zero, new Vector2(0f, 128f));
        GameObject item = CreateUI("Item", content.transform, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image), typeof(Toggle));
        Pin(item.GetComponent<RectTransform>(), new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0.5f, 1f), Vector2.zero, new Vector2(0f, 32f));
        Image itemBackground = item.GetComponent<Image>();
        itemBackground.color = new Color(0.96f, 0.99f, 1f, 1f);
        Toggle itemToggle = item.GetComponent<Toggle>();
        itemToggle.targetGraphic = itemBackground;
        Text itemLabel = CreateText("Item Label", item.transform, "温度（℃）", font, 14, bodyColor, TextAnchor.MiddleLeft, FontStyle.Normal);
        Stretch(itemLabel.rectTransform, Vector2.zero, Vector2.one, new Vector2(12f, 2f), new Vector2(-8f, -2f));
        template.GetComponent<ScrollRect>().content = contentRt;
        template.GetComponent<ScrollRect>().viewport = viewport.GetComponent<RectTransform>();
        dropdown.template = templateRt;
        dropdown.captionText = caption;
        dropdown.itemText = itemLabel;
        dropdown.options.Clear();
        dropdown.options.Add(new Dropdown.OptionData("温度（℃）"));
        dropdown.options.Add(new Dropdown.OptionData("湿度（%）"));
        dropdown.options.Add(new Dropdown.OptionData("CO₂（ppm）"));
        dropdown.options.Add(new Dropdown.OptionData("氨气（ppm）"));
        template.SetActive(false);
        return dropdown;
    }

    private void ShowAbnormalOverlay(string title)
    {
        EnsureAbnormalOverlay();
        abnormalListOverlay.SetActive(true);

        if (abnormalListTitle != null)
        {
            abnormalListTitle.text = title;
        }

        if (abnormalListBody != null)
        {
            abnormalListBody.text = BuildCurrentAbnormalListText();
        }
    }

    private string BuildCurrentAbnormalListText()
    {
        AbnormalVisualizer visualizer = FindObjectOfType<AbnormalVisualizer>();
        List<AbnormalVisualizer.AbnormalItem> items = visualizer != null
            ? visualizer.GetAbnormalItemsSnapshot()
            : null;

        if (items == null || items.Count == 0)
        {
            return "本次任务尚未获取到异常数据。";
        }

        const int maxVisibleRecords = 10;
        int firstIndex = Mathf.Max(0, items.Count - maxVisibleRecords);
        StringBuilder text = new StringBuilder();
        text.Append("本次任务共 ").Append(items.Count).Append(" 条异常\n\n");

        for (int i = items.Count - 1; i >= firstIndex; i--)
        {
            AbnormalVisualizer.AbnormalItem item = items[i];
            if (item == null) continue;

            float confidence = item.abnormalValue <= 1f
                ? item.abnormalValue * 100f
                : item.abnormalValue;
            string location = AbnormalEventActions.FormatCageCode(item.row, item.layer, item.cage);
            string occurTime = string.IsNullOrWhiteSpace(item.occurTime) ? "--" : item.occurTime;

            text.Append('[').Append(string.IsNullOrWhiteSpace(item.abnormalType) ? "其他异常" : item.abnormalType)
                .Append("]  ").Append(location)
                .Append("   置信度：").Append(confidence.ToString("F0")).Append('%')
                .Append("   时间：").Append(occurTime).Append('\n');
        }

        if (items.Count > maxVisibleRecords)
        {
            text.Append("\n仅显示最近 ").Append(maxVisibleRecords).Append(" 条。");
        }

        return text.ToString();
    }

    private void EnsureAbnormalOverlay()
    {
        if (abnormalListOverlay != null)
        {
            return;
        }

        Font font = ResolveFont();

        abnormalListOverlay = CreateUI("AbnormalListOverlay", transform, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        RectTransform overlayRt = abnormalListOverlay.GetComponent<RectTransform>();
        Stretch(overlayRt, Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
        Image overlayImage = abnormalListOverlay.GetComponent<Image>();
        overlayImage.color = overlayColor;
        overlayImage.raycastTarget = true;

        GameObject card = CreateUI("ListCard", abnormalListOverlay.transform, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        RectTransform cardRt = card.GetComponent<RectTransform>();
        Pin(cardRt, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), Vector2.zero, new Vector2(680f, 420f));
        Image cardImage = card.GetComponent<Image>();
        cardImage.color = cardColor;
        Outline outline = card.AddComponent<Outline>();
        outline.effectColor = new Color(0.32f, 0.76f, 0.81f, 0.90f);
        outline.effectDistance = new Vector2(1.2f, -1.2f);

        abnormalListTitle = CreateText("Title", card.transform, "异常列表（本次任务）", font, 24, titleColor, TextAnchor.MiddleLeft, FontStyle.Bold);
        Pin(abnormalListTitle.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0.5f, 1f), new Vector2(0f, -34f), new Vector2(-46f, 38f));

        Image line = CreateImage("AccentLine", card.transform, accentColor);
        Pin(line.rectTransform, new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0.5f, 1f), new Vector2(0f, -62f), new Vector2(-36f, 2f));

        abnormalListBody = CreateText("ListBody", card.transform, "", font, 18, bodyColor, TextAnchor.UpperLeft, FontStyle.Normal);
        abnormalListBody.horizontalOverflow = HorizontalWrapMode.Wrap;
        abnormalListBody.verticalOverflow = VerticalWrapMode.Overflow;
        Stretch(abnormalListBody.rectTransform, new Vector2(0f, 0f), new Vector2(1f, 1f), new Vector2(28f, 54f), new Vector2(-28f, -88f));

        Button close = CreateButton("CloseButton", card.transform, "关闭", redColor, font);
        Pin(close.GetComponent<RectTransform>(), new Vector2(1f, 0f), new Vector2(1f, 0f), new Vector2(1f, 0f), new Vector2(-24f, 20f), new Vector2(94f, 34f));
        close.onClick.AddListener(CloseAbnormalList);

        abnormalListOverlay.SetActive(false);
    }

    private Font ResolveFont()
    {
        Text existing = GetComponentInChildren<Text>(true);
        return existing != null && existing.font != null
            ? existing.font
            : Resources.GetBuiltinResource<Font>("Arial.ttf");
    }

    private static Button CreateButton(string name, Transform parent, string label, Color color, Font font)
    {
        GameObject go = CreateUI(name, parent, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image), typeof(Button));
        Image image = go.GetComponent<Image>();
        image.color = color;
        Button button = go.GetComponent<Button>();

        Text text = CreateText("Text", go.transform, label, font, 15, Color.white, TextAnchor.MiddleCenter, FontStyle.Bold);
        Stretch(text.rectTransform, Vector2.zero, Vector2.one, new Vector2(4f, 2f), new Vector2(-4f, -2f));
        return button;
    }

    private static Text CreateText(string name, Transform parent, string value, Font font, int size, Color color, TextAnchor anchor, FontStyle style)
    {
        GameObject go = CreateUI(name, parent, typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
        Text text = go.GetComponent<Text>();
        text.font = font;
        text.text = value;
        text.fontSize = size;
        text.fontStyle = style;
        text.color = color;
        text.alignment = anchor;
        text.supportRichText = true;
        text.raycastTarget = false;
        return text;
    }

    private static Image CreateImage(string name, Transform parent, Color color)
    {
        GameObject go = CreateUI(name, parent, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        Image image = go.GetComponent<Image>();
        image.color = color;
        image.raycastTarget = false;
        return image;
    }

    private static GameObject CreateUI(string name, Transform parent, params System.Type[] components)
    {
        GameObject go = new GameObject(name, components);
        go.transform.SetParent(parent, false);
        return go;
    }

    private static void SetActive(Transform target, bool state)
    {
        if (target != null)
        {
            target.gameObject.SetActive(state);
        }
    }

    private static void Pin(RectTransform rt, Vector2 anchorMin, Vector2 anchorMax, Vector2 pivot, Vector2 position, Vector2 size)
    {
        rt.anchorMin = anchorMin;
        rt.anchorMax = anchorMax;
        rt.pivot = pivot;
        rt.anchoredPosition = position;
        rt.sizeDelta = size;
    }

    private static void Stretch(RectTransform rt, Vector2 anchorMin, Vector2 anchorMax, Vector2 offsetMin, Vector2 offsetMax)
    {
        rt.anchorMin = anchorMin;
        rt.anchorMax = anchorMax;
        rt.offsetMin = offsetMin;
        rt.offsetMax = offsetMax;
    }
}
