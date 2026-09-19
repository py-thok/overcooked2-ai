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
        /// Restart the current level through the game's own flow (same path as
        /// the pause-menu restart: ServerCampaignFlowController sets a flag and
        /// its GetNextScene reloads the kitchen with a loading screen).
        /// Returns true if the request was delivered.
        /// </summary>
        public static bool RestartLevel(ManualLogSource log)
        {
            ServerCampaignFlowController flow = null;
            try { flow = GameUtils.GetFlowController() as ServerCampaignFlowController; } catch { }
            if (flow == null)
                flow = UnityEngine.Object.FindObjectOfType<ServerCampaignFlowController>();
            if (flow == null)
            {
                if (log != null) log.LogWarning("[OC2Bridge] RESET ignored: no campaign flow (not in a level?)");
                return false;
            }
            flow.OnLevelRestartRequested();
            if (log != null) log.LogInfo("[OC2Bridge] level restart requested");
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
