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
