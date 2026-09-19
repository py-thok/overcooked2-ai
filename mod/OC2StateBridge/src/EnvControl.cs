using BepInEx.Logging;
using UnityEngine;

namespace OC2StateBridge
{
    /// <summary>
    /// Environment control for RL training: level restart and time scaling.
    /// All methods must be called on the Unity main thread.
    /// </summary>
    public static class EnvControl
    {
        /// <summary>Reflection call to internal ServerMessenger.LoadLevel(
        /// sceneName, GameState.LoadKitchen, true, GameState.RunKitchen) —
        /// the pause-menu restart path.</summary>
        private static void InvokeServerLoadLevel(string sceneName)
        {
            MultiplayerController mc = GameUtils.RequireManager<MultiplayerController>();
            if (mc != null) mc.StopSynchronisation();
            System.Type t = typeof(GameUtils).Assembly.GetType("ServerMessenger");
            System.Reflection.MethodInfo m = t.GetMethod("LoadLevel",
                System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic,
                null, new System.Type[] { typeof(string), typeof(GameState), typeof(bool), typeof(GameState) }, null);
            m.Invoke(null, new object[] { sceneName, GameState.LoadKitchen, true, GameState.RunKitchen });
        }

        /// <summary>
        /// Restart the current level immediately, using the same path as the
        /// pause-menu restart (InGamePauseMenu.OnRestartConfirmed):
        /// stop entity synchronisation, then reload the active scene through
        /// ServerMessenger.LoadLevel. Works mid-round, no outro screen.
        /// </summary>
        public static bool RestartLevel(ManualLogSource log)
        {
            try
            {
                MultiplayerController mc = GameUtils.RequireManager<MultiplayerController>();
                if (mc != null) mc.StopSynchronisation();

                // ServerMessenger is internal -> invoke via reflection:
                // LoadLevel(string sceneName, GameState setAtLoadingBegin, bool bUseLoadingScreen, GameState waitForHide)
                string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
                System.Type t = typeof(GameUtils).Assembly.GetType("ServerMessenger");
                System.Reflection.MethodInfo m = t.GetMethod("LoadLevel",
                    System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic,
                    null, new System.Type[] { typeof(string), typeof(GameState), typeof(bool), typeof(GameState) }, null);
                m.Invoke(null, new object[] { scene, GameState.LoadKitchen, true, GameState.RunKitchen });
                if (log != null) log.LogInfo("[OC2Bridge] level restart (LoadLevel) issued");
                return true;
            }
            catch (System.Exception e)
            {
                if (log != null) log.LogWarning("[OC2Bridge] RESET failed: " + e.Message);
                return false;
            }
        }

        /// <summary>
        /// Programmatically pass the StartScreen "press any key" engagement
        /// prompt: engage pad One (keyboard) through PlayerManager, then invoke
        /// StartScreenFlow's private completion handler so the profile loads
        /// and the frontend (main menu) activates — exactly what a real key
        /// press would do. After this, T17FrontendFlow.Instance is available
        /// and BootstrapCampaignSession can run.
        /// </summary>
        public static bool Engage(ManualLogSource log)
        {
            try
            {
                PlayerManager pm = GameUtils.RequireManager<PlayerManager>();
                if (pm.HasPlayer())
                {
                    if (log != null) log.LogInfo("[OC2Bridge] already engaged");
                    return true;
                }
                pm.StartGameownerEngagement(ControlPadInput.PadNum.One, null, delegate (GamepadUser user)
                {
                    if (log != null) log.LogInfo("[OC2Bridge] engagement finished, user=" + (user != null));
                    StartScreenFlow ssr = StartScreenFlow.Instance;
                    if (ssr != null && user != null)
                    {
                        System.Reflection.MethodInfo m = typeof(StartScreenFlow).GetMethod("OnEngagementFinished",
                            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
                        if (m != null) m.Invoke(ssr, new object[] { user });
                    }
                });
                return true;
            }
            catch (System.Exception e)
            {
                if (log != null) log.LogWarning("[OC2Bridge] ENGAGE failed: " + e.Message);
                return false;
            }
        }

        /// <summary>
        /// Replicate the story-mode button flow (SaveSlotElement.ServerLoadCampaign)
        /// so level loads work from the StartScreen without a manual click:
        /// create a fresh coop session, broadcast SetupCoopSession (registers the
        /// session/progress on the client), set campaign mode. Without this the
        /// kitchen loads but chefs are never assigned (black screen, no players).
        /// </summary>
        private static bool BootstrapCampaignSession(ManualLogSource log)
        {
            GameSession session = GameUtils.GetGameSession();
            if (session == null || session.Progress == null
                || session.TypeSettings.Type != GameSession.GameType.Cooperative)
            {
                if (T17FrontendFlow.Instance == null)
                {
                    if (log != null) log.LogWarning("[OC2Bridge] no frontend flow; cannot create session");
                    return false;
                }
                // SelectSaveDialog.CreateFreshGameSessionForSlot(-1, 0)
                // (base-game campaign has DLC id -1, see FrontendCampaignTabOptions):
                session = T17FrontendFlow.Instance.StartEmptySession(GameSession.GameType.Cooperative, -1);
                if (session == null)
                {
                    if (log != null) log.LogWarning("[OC2Bridge] StartEmptySession returned null");
                    return false;
                }
                session.SaveSlot = 0;
                if (log != null) log.LogInfo("[OC2Bridge] created fresh coop session");
            }
            session.FillShownMetaDialogStatus();
            // ServerMessenger.SetupCoopSession(dlc, progressData, metaDialogs, sessionConfig)
            System.Type t = typeof(GameUtils).Assembly.GetType("ServerMessenger");
            System.Reflection.MethodInfo m = t.GetMethod("SetupCoopSession",
                System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);
            m.Invoke(null, new object[] { -1, session.Progress.SaveData, session.m_shownMetaDialogs, session.GameModeSessionConfig });
            ServerGameSetup.Mode = GameMode.Campaign;
            if (log != null) log.LogInfo("[OC2Bridge] coop session broadcast, mode=Campaign");
            return true;
        }

        /// <summary>
        /// Load a kitchen level by scene name, from anywhere (menu, map,
        /// mid-round). Bootstraps a campaign session when none exists
        /// (StartScreen), then resolves the scene directory entry, sets
        /// GameSession.LevelSettings and takes the pause-menu load path.
        /// Returns true if the scene was found.
        /// </summary>
        public static bool LoadLevelBySceneName(string sceneName, ManualLogSource log)
        {
            try
            {
                GameSession session = GameUtils.GetGameSession();
                if (session == null || session.Progress == null
                    || session.TypeSettings.Type != GameSession.GameType.Cooperative)
                {
                    if (!BootstrapCampaignSession(log)) return false;
                    session = GameUtils.GetGameSession();
                }
                else if (ServerGameSetup.Mode != GameMode.Campaign)
                {
                    // session exists but mode was never set (e.g. back to menu)
                    ServerGameSetup.Mode = GameMode.Campaign;
                }
                SceneDirectoryData dir = session.Progress.GetSceneDirectory();
                if (dir == null || dir.Scenes == null)
                {
                    if (log != null) log.LogWarning("[OC2Bridge] LOADLEVEL: no scene directory");
                    return false;
                }
                for (int i = 0; i < dir.Scenes.Length; i++)
                {
                    SceneDirectoryData.SceneDirectoryEntry entry = dir.Scenes[i];
                    for (int v = 0; v < entry.SceneVarients.Length; v++)
                    {
                        SceneDirectoryData.PerPlayerCountDirectoryEntry variant = entry.SceneVarients[v];
                        if (variant == null || variant.SceneName != sceneName) continue;

                        GameSession.GameLevelSettings settings = new GameSession.GameLevelSettings();
                        settings.SceneDirectoryVarientEntry = variant;
                        session.LevelSettings = settings;
                        session.Progress.SaveData.LastLevelEntered = i;
                        // Use the pause-menu-restart path (ServerMessenger.LoadLevel):
                        // it re-runs the full multiplayer/entity setup that
                        // LoadingScreenFlow.LoadScene alone skips.
                        InvokeServerLoadLevel(sceneName);
                        if (log != null) log.LogInfo("[OC2Bridge] loading level " + sceneName + " (dir index " + i + ", variant " + v + ")");
                        return true;
                    }
                }
                if (log != null) log.LogWarning("[OC2Bridge] scene not in directory: " + sceneName);
                return false;
            }
            catch (System.Exception e)
            {
                if (log != null) log.LogWarning("[OC2Bridge] LOADLEVEL failed: " + e.Message);
                return false;
            }
        }

        /// <summary>Teleport a player (by sorted slot index) to a world position
        /// and zero its velocity. For calibration and training scenario setup.</summary>
        public static bool SetPlayerPos(int player, Vector3 pos, ManualLogSource log)
        {
            GameObject[] players = GameObject.FindGameObjectsWithTag("Player");
            System.Array.Sort(players, delegate (GameObject a, GameObject b)
            {
                return string.CompareOrdinal(a.name, b.name);
            });
            if (player < 0 || player >= players.Length) return false;
            Rigidbody rb = players[player].GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.velocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
                rb.position = pos;
            }
            players[player].transform.position = pos;
            return true;
        }

        private static float _desiredTimeScale = 1f;

        /// <summary>Scale game speed for training. Clamped to [0.25, 8].
        /// The game re-writes Time.timeScale every frame (TimeManager.Update,
        /// GameDebugManager.Update), so the value is pinned in LateUpdate.</summary>
        public static float SetTimeScale(float scale, ManualLogSource log)
        {
            scale = Mathf.Clamp(scale, 0.25f, 8f);
            _desiredTimeScale = scale;
            ApplyTimeScale();
            if (log != null) log.LogInfo("[OC2Bridge] timeScale = " + scale);
            return scale;
        }

        /// <summary>Called from Plugin.LateUpdate so our scale wins over the
        /// game's own per-frame assignments. No-op while scale is 1.</summary>
        public static void ApplyTimeScale()
        {
            if (_desiredTimeScale == 1f) return;
            Time.timeScale = _desiredTimeScale;
            Time.fixedDeltaTime = 0.02f * _desiredTimeScale;
        }

        public static float GetTimeScale()
        {
            return _desiredTimeScale;
        }
    }
}
