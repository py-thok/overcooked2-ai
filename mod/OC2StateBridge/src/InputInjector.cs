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

                scheme.m_moveX = slot.MoveX;
                scheme.m_moveY = slot.MoveY;
                scheme.m_pickupButton = slot.Pickup;
                scheme.m_worksurfaceUseButton = slot.Use;
                scheme.m_dashButton = slot.Dash;
                slot.HookedScheme = scheme;
                if (log != null)
                    log.LogInfo("[OC2Bridge] input hooked for player slot " + i + " (" + players[i].name + ")");
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
                    slot.MoveX.Value = Convert.ToSingle(mv[0]);
                    slot.MoveY.Value = Convert.ToSingle(mv[1]);
                }
            }
            if (d.ContainsKey("pickup")) slot.Pickup.SetDown(Convert.ToBoolean(d["pickup"]));
            if (d.ContainsKey("use")) slot.Use.SetDown(Convert.ToBoolean(d["use"]));
            if (d.ContainsKey("dash")) slot.Dash.SetDown(Convert.ToBoolean(d["dash"]));
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
