using System;
using System.Collections.Generic;
using BepInEx.Logging;
using UnityEngine;

namespace OC2StateBridge
{
    /// <summary>
    /// Drives players by swapping their PlayerControls.ControlSchemeData input
    /// slots (public fields: m_moveX/m_moveY ILogicalValue, m_pickupButton /
    /// m_worksurfaceUseButton / m_dashButton ILogicalButton) for plugin-owned
    /// implementations. All reads and writes happen on the main thread
    /// (actions are applied from Plugin.Update via StateServer.DrainActions).
    /// </summary>
    public static class InputInjector
    {
        private class Slot
        {
            // Preferred path: drive the game's own network input wrappers
            public NetworkLogicalValue NetMoveX;
            public NetworkLogicalValue NetMoveY;
            public NetworkLogicalButton NetPickup;
            public NetworkLogicalButton NetUse;
            public NetworkLogicalButton NetDash;
            // Fallback path: plugin-owned replacements
            public PluginLogicalValue MoveX = new PluginLogicalValue();
            public PluginLogicalValue MoveY = new PluginLogicalValue();
            public PluginLogicalButton Pickup = new PluginLogicalButton();
            public PluginLogicalButton Use = new PluginLogicalButton();
            public PluginLogicalButton Dash = new PluginLogicalButton();
            public PlayerControls.ControlSchemeData HookedScheme;
        }

        private static readonly List<Slot> _slots = new List<Slot>();
        private static float _nextScan;

        /// <summary>Called every frame from Plugin.Update.</summary>
        public static void EnsureHooked(ManualLogSource log)
        {
            if (Time.time < _nextScan) return;
            _nextScan = Time.time + 2f;

            GameObject[] players;
            try { players = GameObject.FindGameObjectsWithTag("Player"); }
            catch { return; }
            Array.Sort(players, delegate (GameObject a, GameObject b)
            {
                return string.CompareOrdinal(a.name, b.name);
            });

            for (int i = 0; i < players.Length; i++)
            {
                PlayerControls pc = players[i].GetComponent<PlayerControls>();
                if (pc == null) continue;
                PlayerControls.ControlSchemeData scheme = pc.ControlScheme;
                if (scheme == null) continue; // not assigned yet, retry next scan

                while (_slots.Count <= i) _slots.Add(new Slot());
                Slot slot = _slots[i];
                if (ReferenceEquals(slot.HookedScheme, scheme)) continue;

                // The server wraps input slots with NetworkLogical* (see
                // ServerInputReceiver). Driving those keeps the game's input
                // pipeline intact; only replace slots when wrappers are absent.
                slot.NetMoveX = scheme.m_moveX as NetworkLogicalValue;
                slot.NetMoveY = scheme.m_moveY as NetworkLogicalValue;
                slot.NetPickup = scheme.m_pickupButton as NetworkLogicalButton;
                slot.NetUse = scheme.m_worksurfaceUseButton as NetworkLogicalButton;
                slot.NetDash = scheme.m_dashButton as NetworkLogicalButton;

                if (slot.NetMoveX == null) scheme.m_moveX = slot.MoveX;
                if (slot.NetMoveY == null) scheme.m_moveY = slot.MoveY;
                if (slot.NetPickup == null) scheme.m_pickupButton = slot.Pickup;
                if (slot.NetUse == null) scheme.m_worksurfaceUseButton = slot.Use;
                if (slot.NetDash == null) scheme.m_dashButton = slot.Dash;

                slot.HookedScheme = scheme;
                if (log != null)
                    log.LogInfo("[OC2Bridge] input hooked for player slot " + i + " (" + players[i].name +
                        ") network-wrappers=" + (slot.NetMoveX != null));
            }
        }

        /// <summary>
        /// Apply one action JSON line (main thread).
        /// {"player":0, "move":[x,y], "pickup":true, "use":false, "dash":true}
        /// Absent keys leave that channel unchanged.
        /// </summary>
        public static void ApplyAction(string json)
        {
            Dictionary<string, object> d = JsonParser.ParseObject(json);
            if (d == null) return;

            int player = d.ContainsKey("player") ? Convert.ToInt32(d["player"]) : 0;
            if (player < 0 || player >= _slots.Count) return;
            Slot slot = _slots[player];

            if (d.ContainsKey("move"))
            {
                List<object> mv = d["move"] as List<object>;
                if (mv != null && mv.Count >= 2)
                {
                    float x = Convert.ToSingle(mv[0]);
                    float y = Convert.ToSingle(mv[1]);
                    if (slot.NetMoveX != null) slot.NetMoveX.SetValue(x); else slot.MoveX.Value = x;
                    if (slot.NetMoveY != null) slot.NetMoveY.SetValue(y); else slot.MoveY.Value = y;
                }
            }
            if (d.ContainsKey("pickup")) SetButton(slot, 0, Convert.ToBoolean(d["pickup"]));
            if (d.ContainsKey("use")) SetButton(slot, 1, Convert.ToBoolean(d["use"]));
            if (d.ContainsKey("dash")) SetButton(slot, 2, Convert.ToBoolean(d["dash"]));
        }

        private static void SetButton(Slot slot, int which, bool down)
        {
            NetworkLogicalButton net = which == 0 ? slot.NetPickup : which == 1 ? slot.NetUse : slot.NetDash;
            PluginLogicalButton own = which == 0 ? slot.Pickup : which == 1 ? slot.Use : slot.Dash;
            if (net != null) net.SetIsDown(down);
            else own.SetDown(down);
        }
    }

    /// <summary>Plugin-owned analog axis.</summary>
    public class PluginLogicalValue : ILogicalValue
    {
        public float Value;

        public float GetValue() { return Value; }

        public void GetLogicTreeData(out AcyclicGraph<ILogicalElement, LogicalLinkInfo> _tree,
            out AcyclicGraph<ILogicalElement, LogicalLinkInfo>.Node _head)
        {
            _tree = new AcyclicGraph<ILogicalElement, LogicalLinkInfo>(this);
            _head = _tree.GetNode(this);
        }
    }

    /// <summary>Plugin-owned digital button with edge events.</summary>
    public class PluginLogicalButton : ILogicalButton
    {
        private bool _down;
        private bool _justPressed;
        private bool _justReleased;
        private float _downSince;

        public void SetDown(bool down)
        {
            if (down == _down) return;
            _down = down;
            if (down) { _justPressed = true; _downSince = Time.time; }
            else { _justReleased = true; }
        }

        public bool IsDown() { return _down; }

        public bool JustPressed()
        {
            bool v = _justPressed;
            _justPressed = false;
            return v;
        }

        public bool JustReleased()
        {
            bool v = _justReleased;
            _justReleased = false;
            return v;
        }

        public bool HasUnclaimedPressEvent() { return _justPressed; }
        public void ClaimPressEvent() { _justPressed = false; }
        public bool HasUnclaimedReleaseEvent() { return _justReleased; }
        public void ClaimReleaseEvent() { _justReleased = false; }

        public float GetHeldTimeLength()
        {
            return _down ? Time.time - _downSince : 0f;
        }

        public void GetLogicTreeData(out AcyclicGraph<ILogicalElement, LogicalLinkInfo> _tree,
            out AcyclicGraph<ILogicalElement, LogicalLinkInfo>.Node _head)
        {
            _tree = new AcyclicGraph<ILogicalElement, LogicalLinkInfo>(this);
            _head = _tree.GetNode(this);
        }
    }
}
