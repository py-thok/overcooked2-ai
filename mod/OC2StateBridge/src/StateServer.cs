using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using BepInEx.Logging;

namespace OC2StateBridge
{
    /// <summary>
    /// Single-client TCP server. The game thread publishes the latest state
    /// snapshot; client threads only read the published string, so no game
    /// API is ever touched off the main thread. Action lines are queued and
    /// drained by the main thread (Plugin.Update).
    /// </summary>
    public class StateServer
    {
        private readonly int _port;
        private readonly ManualLogSource _log;
        private TcpListener _listener;
        private Thread _acceptThread;
        private volatile bool _running;

        private readonly object _stateLock = new object();
        private string _latestState = "{}";

        private readonly object _actionLock = new object();
        private readonly Queue<string> _pendingActions = new Queue<string>();

        public StateServer(int port, ManualLogSource log)
        {
            _port = port;
            _log = log;
        }

        public void Start()
        {
            _running = true;
            _acceptThread = new Thread(AcceptLoop);
            _acceptThread.IsBackground = true;
            _acceptThread.Start();
        }

        public void Stop()
        {
            _running = false;
            try { if (_listener != null) _listener.Stop(); } catch { }
        }

        public void PublishState(string json)
        {
            lock (_stateLock) { _latestState = json; }
        }

        /// <summary>Called on the main thread; applies every queued action line.</summary>
        public void DrainActions(Action<string> apply)
        {
            lock (_actionLock)
            {
                while (_pendingActions.Count > 0)
                {
                    string line = _pendingActions.Dequeue();
                    try { apply(line); }
                    catch (Exception e) { _log.LogError("[OC2Bridge] action apply failed: " + e.Message); }
                }
            }
        }

        private void AcceptLoop()
        {
            try
            {
                _listener = new TcpListener(IPAddress.Loopback, _port);
                _listener.Start();
            }
            catch (Exception e)
            {
                _log.LogError("[OC2Bridge] failed to bind port " + _port + ": " + e.Message);
                return;
            }
            _log.LogInfo("[OC2Bridge] listening on 127.0.0.1:" + _port);

            while (_running)
            {
                TcpClient client;
                try { client = _listener.AcceptTcpClient(); }
                catch { break; } // listener stopped
                _log.LogInfo("[OC2Bridge] client connected");
                Thread t = new Thread(delegate () { HandleClient(client); });
                t.IsBackground = true;
                t.Start();
            }
        }

        private void HandleClient(TcpClient client)
        {
            try
            {
                using (client)
                using (NetworkStream stream = client.GetStream())
                {
                    // UTF8 without BOM: a BOM on the first reply would corrupt client parsing
                    UTF8Encoding enc = new UTF8Encoding(false);
                    StreamReader reader = new StreamReader(stream, enc);
                    StreamWriter writer = new StreamWriter(stream, enc);
                    writer.AutoFlush = true;

                    string line;
                    while (_running && (line = reader.ReadLine()) != null)
                    {
                        if (line == "GET")
                        {
                            string snapshot;
                            lock (_stateLock) { snapshot = _latestState; }
                            writer.WriteLine(snapshot);
                        }
                        else if (line.StartsWith("ACTION "))
                        {
                            lock (_actionLock) { _pendingActions.Enqueue(line.Substring(7)); }
                            writer.WriteLine("OK");
                        }
                        else if (line == "RESET")
                        {
                            lock (_actionLock) { _pendingActions.Enqueue("@RESET"); }
                            writer.WriteLine("OK");
                        }
                        else if (line.StartsWith("TIMESCALE "))
                        {
                            lock (_actionLock) { _pendingActions.Enqueue("@" + line); }
                            writer.WriteLine("OK");
                        }
                        else if (line == "PING")
                        {
                            writer.WriteLine("PONG");
                        }
                        else
                        {
                            writer.WriteLine("ERR unknown command");
                        }
                    }
                }
            }
            catch (Exception e)
            {
                _log.LogInfo("[OC2Bridge] client disconnected: " + e.Message);
            }
        }
    }
}
