using BepInEx.Logging;
using UnityEngine;

namespace OC2StateBridge
{
    /// <summary>
    /// Keeps cooked pots permanently fresh for training: a pot burns
    /// (progress > 2x cook time) ~30 game-seconds after cooking finishes,
    /// which left every episode with a single tiny scoop window and a dead
    /// pot for the remaining ~150s (observed: is_burning=True for the whole
    /// back half of every round). Clamping progress to cookTime once the pot
    /// is done keeps the scoop window open all episode.
    ///
    /// NOTE: this disables burn discipline by design. Revisit before
    /// generalizing — a policy trained this way never learns to serve
    /// before the burn, nor to recover a ruined pot.
    /// </summary>
    public static class Janitor
    {
        public static bool Enabled = true;

        public static void KeepPotsFresh(ManualLogSource log)
        {
            if (!Enabled) return;
            foreach (ServerCookingHandler h in ServerCookingHandler.GetCookingHandlers())
            {
                if (h == null) continue;
                try
                {
                    float cookTime = h.AccessCookingTime;
                    if (cookTime <= 0f) continue;
                    if (h.GetCookingProgress() <= 1.2f * cookTime) continue;
                    // only fresh-keep pots that actually hold food
                    IIngredientContents ic = h.gameObject.RequestInterface<IIngredientContents>();
                    AssembledDefinitionNode[] contents = ic != null ? ic.GetContents() : null;
                    if (contents == null || contents.Length == 0) continue;
                    h.SetCookingProgress(cookTime);
                }
                catch { }
            }
        }
    }
}
