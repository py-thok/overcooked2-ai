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

        private void Awake()
        {
            Logger.LogInfo("[OC2Bridge] awake, starting state server on 127.0.0.1:8765");
            _server = new StateServer(8765, Logger);
            _server.Start();
        }

        private void Update()
        {
            // GameObject APIs must run on the main thread -> collect here,
            // the network thread only serves the latest snapshot string.
            string json = StateExtractor.Collect();
            _server.PublishState(json);

            InputInjector.EnsureHooked(Logger);
            _server.DrainActions(InputInjector.ApplyAction);

            if (Input.GetKeyDown(KeyCode.F9))
            {
                Discovery.Dump(Logger);
            }
        }

        private void OnDestroy()
        {
            if (_server != null) _server.Stop();
        }
    }
}
