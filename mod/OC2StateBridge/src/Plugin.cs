using BepInEx;
using BepInEx.Logging;
using UnityEngine;

namespace OC2StateBridge
{
    /// <summary>
    /// BepInEx entry point. Every frame (main thread) it collects the full
    /// kitchen state, hands it to the TCP state server, and keeps the input
    /// injector hooked onto the bot-controlled player.
    ///
    /// Protocol (newline-delimited text on 127.0.0.1:8765):
    ///   C->S "GET"            -> S->C one line of JSON state
    ///   C->S "ACTION {json}"  -> S->C "OK"   (json: {"move":[x,y],"pickup":bool,"use":bool,"dash":bool,"player":0})
    ///   C->S "PING"           -> S->C "PONG"
    /// Press F9 in-game to dump a discovery report (component/type census) to
    /// <game dir>\oc2bridge_discovery.log for validating field assumptions.
    /// </summary>
    [BepInPlugin("com.oc2.statebridge", "OC2 State Bridge", "0.1.0")]
    public class Plugin : BaseUnityPlugin
    {
        private StateServer _server;
        private float _nextDebugLog;

        private void Awake()
        {
            // Port override lets a second game instance run its own bridge
            // (parallel RL): OC2BRIDGE_PORT=8766 Overcooked2.exe
            int port = 8765;
            string envPort = System.Environment.GetEnvironmentVariable("OC2BRIDGE_PORT");
            if (!string.IsNullOrEmpty(envPort)) int.TryParse(envPort, out port);
            Logger.LogInfo("[OC2Bridge] awake, starting state server on 127.0.0.1:" + port);
            _server = new StateServer(port, Logger);
            _server.Start();
        }

        private void Update()
        {
            // GameObject APIs must run on the main thread -> collect here,
            // the network thread only serves the latest snapshot string.
            string json = StateExtractor.Collect();
            _server.PublishState(json);
            _server.PublishStations(StateExtractor.StationsJson);

            InputInjector.EnsureHooked(Logger);
            _server.DrainActions(DispatchCommand);
            EnvControl.TickPendingLoad(Logger);

            if (Input.GetKeyDown(KeyCode.F9))
            {
                Discovery.Dump(Logger);
            }

            if (Time.time >= _nextDebugLog)
            {
                _nextDebugLog = Time.time + 2f;
                Logger.LogInfo("[OC2Bridge] btn reads: JustPressed true/calls=" +
                    PluginLogicalButton.s_justPressedTrue + "/" + PluginLogicalButton.s_justPressedCalls +
                    " IsDown true/calls=" + PluginLogicalButton.s_isDownTrue + "/" + PluginLogicalButton.s_isDownCalls);
            }
        }

        /// <summary>Main-thread dispatch of queued client commands.</summary>
        private void DispatchCommand(string cmd)
        {
            if (cmd == "@RESET")
            {
                EnvControl.RestartLevel(Logger);
            }
            else if (cmd == "@ENGAGE")
            {
                EnvControl.Engage(Logger);
            }
            else if (cmd.StartsWith("@TIMESCALE "))
            {
                float scale;
                if (float.TryParse(cmd.Substring(11),
                        System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out scale))
                    EnvControl.SetTimeScale(scale, Logger);
            }
            else if (cmd.StartsWith("@LOADLEVEL "))
            {
                EnvControl.LoadLevelBySceneName(cmd.Substring(11).Trim(), Logger);
            }
            else if (cmd.StartsWith("@MODE "))
            {
                InputInjector.SetMode(cmd.Substring(6).Trim() == "drive", Logger);
            }
            else if (cmd.StartsWith("@INTERACT "))
            {
                int p = int.Parse(cmd.Substring(10).Trim());
                Logger.LogInfo("[OC2Bridge] interact: " + SemanticActions.Interact(p, Logger));
            }
            else if (cmd.StartsWith("@THROW "))
            {
                int p = int.Parse(cmd.Substring(7).Trim());
                Logger.LogInfo("[OC2Bridge] throw: " + SemanticActions.ThrowItem(p, Logger));
            }
            else if (cmd.StartsWith("@FETCHCRATE "))
            {
                // @FETCHCRATE <player> <x> <z>
                string[] parts = cmd.Substring(12).Split(' ');
                Logger.LogInfo("[OC2Bridge] fetchcrate: " + SemanticActions.FetchCrate(
                    int.Parse(parts[0]),
                    float.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture),
                    float.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture), Logger));
            }
            else if (cmd.StartsWith("@PLACEBOARD "))
            {
                // @PLACEBOARD <player> <x> <z>
                string[] parts = cmd.Substring(12).Split(' ');
                Logger.LogInfo("[OC2Bridge] placeboard: " + SemanticActions.PlaceOnBoard(
                    int.Parse(parts[0]),
                    float.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture),
                    float.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture), Logger));
            }
            else if (cmd.StartsWith("@SETPOS "))
            {
                // @SETPOS <player> <x> <y> <z>
                string[] parts = cmd.Substring(8).Split(' ');
                if (parts.Length == 4)
                {
                    int p = int.Parse(parts[0]);
                    UnityEngine.Vector3 pos = new UnityEngine.Vector3(
                        float.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture),
                        float.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture),
                        float.Parse(parts[3], System.Globalization.CultureInfo.InvariantCulture));
                    EnvControl.SetPlayerPos(p, pos, Logger);
                }
            }
            else
            {
                InputInjector.ApplyAction(cmd);
            }
        }

        private void OnDestroy()
        {
            if (_server != null) _server.Stop();
        }

        private void LateUpdate()
        {
            // runs after the game's own Update-pass timeScale writes
            EnvControl.ApplyTimeScale();
        }
    }
}
