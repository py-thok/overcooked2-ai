using System;
using System.IO;
using System.Text;
using BepInEx.Logging;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace OC2StateBridge
{
    /// <summary>
    /// F9-triggered census of live game objects: validates that the classes
    /// and access paths used by StateExtractor actually resolve in the running
    /// game. Writes <game dir>\oc2bridge_discovery.log.
    /// </summary>
    public static class Discovery
    {
        public static void Dump(ManualLogSource log)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("=== OC2Bridge discovery " + DateTime.Now.ToString("s") + " ===");
            sb.AppendLine("scene: " + SceneManager.GetActiveScene().name);

            // Flow controller
            IFlowController f = null;
            try { f = GameUtils.GetFlowController(); } catch (Exception e) { sb.AppendLine("GetFlowController threw: " + e.Message); }
            sb.AppendLine("flow: " + (f == null ? "null" : f.GetType().FullName));
            ServerKitchenFlowControllerBase sf = f as ServerKitchenFlowControllerBase;
            if (sf == null) sf = UnityEngine.Object.FindObjectOfType<ServerKitchenFlowControllerBase>();
            sb.AppendLine("server flow: " + (sf == null ? "null" : sf.GetType().FullName));
            if (sf != null)
            {
                object timer = null;
                try { timer = sf.RoundTimer; } catch { }
                sb.AppendLine("  round timer: " + (timer == null ? "null" : timer.GetType().FullName));
                try
                {
                    ServerTeamMonitor mon = sf.GetMonitorForTeam(TeamID.One);
                    sb.AppendLine("  team monitor: " + (mon == null ? "null" : "ok"));
                    if (mon != null)
                        sb.AppendLine("  orders controller: " +
                            (mon.OrdersController == null ? "null" : mon.OrdersController.GetType().FullName));
                }
                catch (Exception e) { sb.AppendLine("  monitor error: " + e.Message); }
            }

            // Grid
            sb.AppendLine("grid managers active: " + GridManager.GetActiveCount());
            if (GridManager.GetActiveCount() > 0)
            {
                GridManager g = GridManager.GetActive(0);
                Point3 hs = g.AccessGridHalfSize;
                sb.AppendLine("  grid[0] " + g.GetType().Name + " half_size=" + hs.X + "," + hs.Y + "," + hs.Z);
            }

            // Players
            GameObject[] players = GameObject.FindGameObjectsWithTag("Player");
            sb.AppendLine("players: " + players.Length);
            foreach (GameObject p in players)
            {
                sb.AppendLine("  " + p.name + " pos=" + p.transform.position);
                PlayerControls pc = p.GetComponent<PlayerControls>();
                sb.AppendLine("    PlayerControls: " + (pc != null) +
                    " scheme: " + (pc != null && pc.ControlScheme != null ? pc.ControlScheme.Player.ToString() : "null"));
                Component carrier = p.GetComponent(typeof(IPlayerCarrier)) as Component;
                sb.AppendLine("    IPlayerCarrier: " + (carrier != null ? carrier.GetType().Name : "null"));
                if (carrier != null)
                {
                    GameObject held = null;
                    try { held = ((IPlayerCarrier)carrier).InspectCarriedItem(PlayerAttachTarget.Default); } catch { }
                    sb.AppendLine("    held: " + (held == null ? "nothing" : DescribeGo(held)));
                }
                foreach (Component c in p.GetComponents<Component>())
                    if (c != null) sb.AppendLine("    cmp: " + c.GetType().Name);
            }

            // Cooking handlers
            int n = 0;
            foreach (ServerCookingHandler h in ServerCookingHandler.GetCookingHandlers())
            {
                if (h == null) continue;
                n++;
                sb.AppendLine("cooker[" + n + "]: " + h.gameObject.name + " pos=" + h.transform.position +
                    " progress=" + h.GetCookingProgress() + "/" + h.AccessCookingTime +
                    " cooked=" + h.IsCooked() + " burning=" + h.IsBurning());
                foreach (Component c in h.gameObject.GetComponents<Component>())
                    if (c != null) sb.AppendLine("    cmp: " + c.GetType().Name);
            }
            if (n == 0) sb.AppendLine("cookers: none found");

            string path = Path.Combine(Directory.GetCurrentDirectory(), "oc2bridge_discovery.log");
            File.WriteAllText(path, sb.ToString());
            if (log != null) log.LogInfo("[OC2Bridge] discovery written to " + path);
        }

        private static string DescribeGo(GameObject go)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(go.name).Append(" [");
            Component[] comps = go.GetComponents<Component>();
            for (int i = 0; i < comps.Length; i++)
            {
                if (comps[i] == null) continue;
                if (i > 0) sb.Append(", ");
                sb.Append(comps[i].GetType().Name);
            }
            return sb.Append("]").ToString();
        }
    }
}
