using System;
using System.Collections;
using System.Reflection;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace OC2StateBridge
{
    /// <summary>
    /// Builds the JSON state snapshot from live game objects.
    /// MUST be called from the Unity main thread only.
    ///
    /// Sources (verified against decompiled Assembly-CSharp):
    ///   players  : GameObject tag "Player" -> PlayerControls + IPlayerCarrier
    ///   grid     : GridManager.GetActive(0).GetGridLocationFromPos
    ///   cookers  : ServerCookingHandler.GetCookingHandlers()
    ///   orders   : ServerKitchenFlowControllerBase -> ServerTeamMonitor.OrdersController
    ///              (private m_activeOrders via reflection)
    ///   score    : ServerTeamMonitor.Score (TotalBaseScore + TotalTipsScore)
    ///   timer    : ServerKitchenFlowControllerBase.RoundTimer (TimeElapsed,
    ///              private m_timeLimit via reflection)
    /// </summary>
    public static class StateExtractor
    {
        private static GameObject[] _players = new GameObject[0];
        private static float _nextPlayerScan;
        private static GridManager _grid;
        private static float _nextGridScan;
        private static ServerCookingHandler[] _cookers = new ServerCookingHandler[0];
        private static float _nextCookerScan;
        private static PhysicalAttachment[] _items = new PhysicalAttachment[0];
        private static float _nextItemScan;
        private static int _seq;
        private static string _stationsJson = "{}";
        private static float _nextStationScan;

        /// <summary>Latest station census JSON (refreshed in Collect).</summary>
        public static string StationsJson { get { return _stationsJson; } }

        public static string Collect()
        {
            _seq++;
            StringBuilder sb = new StringBuilder(4096);
            JsonWriter w = new JsonWriter(sb);
            w.BeginObject();
            w.Key("seq"); w.Value(_seq);
            w.Key("t"); w.Value(Time.time);
            w.Key("scene"); w.Value(SceneManager.GetActiveScene().name);

            ServerKitchenFlowControllerBase flow = FindFlow();
            w.Key("in_round"); w.Value(flow != null);

            WriteRound(w, flow);
            WritePlayers(w);
            WriteCookers(w);
            WriteItems(w);
            WriteOrders(w, flow);

            w.EndObject();

            if (Time.time >= _nextStationScan)
            {
                _nextStationScan = Time.time + 2f;
                try { _stationsJson = CollectStations(); } catch { }
            }
            return sb.ToString();
        }

        // ---------------------------------------------------------------
        private static ServerKitchenFlowControllerBase FindFlow()
        {
            IFlowController f = null;
            try { f = GameUtils.GetFlowController(); } catch { }
            ServerKitchenFlowControllerBase s = f as ServerKitchenFlowControllerBase;
            if (s == null)
            {
                // fallback: single-player always has an in-process server controller
                s = UnityEngine.Object.FindObjectOfType<ServerKitchenFlowControllerBase>();
            }
            return s;
        }

        private static void WriteRound(JsonWriter w, ServerKitchenFlowControllerBase flow)
        {
            w.Key("round");
            w.BeginObject();
            float elapsed = -1f, limit = -1f;
            int baseScore = 0, tips = 0;
            if (flow != null)
            {
                object timer = flow.RoundTimer;
                if (timer != null)
                {
                    object e = GetProp(timer, "TimeElapsed");
                    if (e is float) elapsed = (float)e;
                    object l = GetField(timer, "m_timeLimit");
                    if (l is float) limit = (float)l;
                }
                try
                {
                    ServerTeamMonitor mon = flow.GetMonitorForTeam(TeamID.One);
                    if (mon != null && mon.Score != null)
                    {
                        baseScore = mon.Score.TotalBaseScore;
                        tips = mon.Score.TotalTipsScore;
                    }
                }
                catch { }
            }
            w.Key("time_elapsed"); w.Value(elapsed);
            w.Key("time_limit"); w.Value(limit);
            w.Key("time_remaining"); w.Value(limit >= 0f ? Mathf.Max(limit - elapsed, 0f) : -1f);
            w.Key("score"); w.Value(baseScore + tips);
            w.Key("tips"); w.Value(tips);
            w.EndObject();
        }

        private static void WritePlayers(JsonWriter w)
        {
            if (Time.time >= _nextPlayerScan)
            {
                _nextPlayerScan = Time.time + 2f;
                try { _players = GameObject.FindGameObjectsWithTag("Player"); }
                catch { _players = new GameObject[0]; }
                Array.Sort(_players, delegate (GameObject a, GameObject b)
                {
                    return string.CompareOrdinal(a.name, b.name);
                });
            }
            if (Time.time >= _nextGridScan)
            {
                _nextGridScan = Time.time + 5f;
                _grid = GridManager.GetActiveCount() > 0 ? GridManager.GetActive(0) : null;
            }

            w.Key("grid");
            w.BeginObject();
            w.Key("active"); w.Value(_grid != null);
            if (_grid != null)
            {
                try
                {
                    Point3 hs = _grid.AccessGridHalfSize;
                    w.Key("half_size"); w.BeginArray();
                    w.Value(hs.X); w.Value(hs.Y); w.Value(hs.Z);
                    w.EndArray();
                }
                catch { w.Key("half_size"); w.Null(); }
            }
            else { w.Key("half_size"); w.Null(); }
            w.EndObject();

            w.Key("players");
            w.BeginArray();
            for (int i = 0; i < _players.Length; i++)
            {
                GameObject go = _players[i];
                if (go == null) continue;
                w.BeginObject();
                w.Key("i"); w.Value(i);
                w.Key("name"); w.Value(go.name);

                Vector3 pos = go.transform.position;
                Vector3 fwd = go.transform.forward;
                w.Key("pos"); WriteVec3(w, pos);
                w.Key("fwd"); WriteVec3(w, fwd);
                w.Key("grid"); WriteGridIndex(w, pos);

                PlayerControls pc = go.GetComponent<PlayerControls>();
                int playerId = -1;
                if (pc != null && pc.ControlScheme != null)
                    playerId = (int)pc.ControlScheme.Player;
                w.Key("player_id"); w.Value(playerId);

                // Movement input maps directly to world axes:
                //   world_dir = (signX * inputX, -signY * inputY)
                // (PlayerControlsHelper.GetControlAxis). Expose the per-level
                // inversion signs so clients can convert directions exactly.
                int signX = 1, signY = 1;
                if (pc != null)
                {
                    object mov = GetField(pc, "m_movement");
                    if (mov != null)
                    {
                        object ix = GetField(mov, "XAxisAllignment");
                        object iy = GetField(mov, "YAxisAllignment");
                        if (ix != null && (int)ix != 0) signX = -1;
                        if (iy != null && (int)iy != 0) signY = -1;
                    }
                }
                w.Key("move_sign"); w.BeginArray(); w.Value(signX); w.Value(signY); w.EndArray();

                // Echo current inputs (human in sniff mode, bot in drive mode)
                w.Key("input");
                if (pc != null && pc.ControlScheme != null)
                {
                    PlayerControls.ControlSchemeData cs = pc.ControlScheme;
                    w.BeginObject();
                    w.Key("move"); w.BeginArray();
                    w.Value(cs.m_moveX != null ? cs.m_moveX.GetValue() : 0f);
                    w.Value(cs.m_moveY != null ? cs.m_moveY.GetValue() : 0f);
                    w.EndArray();
                    w.Key("pickup"); w.Value(cs.m_pickupButton != null && cs.m_pickupButton.IsDown());
                    w.Key("use"); w.Value(cs.m_worksurfaceUseButton != null && cs.m_worksurfaceUseButton.IsDown());
                    w.Key("dash"); w.Value(cs.m_dashButton != null && cs.m_dashButton.IsDown());
                    w.EndObject();
                }
                else w.Null();

                w.Key("held");
                IPlayerCarrier carrier = (IPlayerCarrier)go.GetComponent(typeof(IPlayerCarrier));
                GameObject held = null;
                if (carrier != null)
                {
                    try { held = carrier.InspectCarriedItem(PlayerAttachTarget.Default); } catch { }
                }
                if (held == null) w.Null();
                else WriteItem(w, held);
                w.EndObject();
            }
            w.EndArray();
        }

        private static void WriteItems(JsonWriter w)
        {
            // Loose/carried items: scan PhysicalAttachment objects periodically.
            // Held items are reported per-player already; this covers items on
            // counters / floor / stations.
            if (Time.time >= _nextItemScan)
            {
                _nextItemScan = Time.time + 0.3f;
                try { _items = UnityEngine.Object.FindObjectsOfType<PhysicalAttachment>(); }
                catch { _items = new PhysicalAttachment[0]; }
            }
            w.Key("items");
            w.BeginArray();
            foreach (PhysicalAttachment pa in _items)
            {
                if (pa == null) continue;
                GameObject go = pa.gameObject;
                w.BeginObject();
                w.Key("name"); w.Value(CleanName(go.name));
                Vector3 pos = go.transform.position;
                w.Key("pos"); WriteVec3(w, pos);
                w.Key("grid"); WriteGridIndex(w, pos);
                w.Key("kind"); w.Value(ClassifyItem(go));
                w.EndObject();
            }
            w.EndArray();
        }

        private static string ClassifyItem(GameObject go)
        {
            if (go.GetComponent<Plate>() != null) return "plate";
            if (go.GetComponent<CookableContainer>() != null) return "utensil";
            if (go.GetComponent<CarryableItem>() != null) return "carryable";
            return "object";
        }

        /// <summary>One-shot census of attach stations (STATIONS command).</summary>
        public static string CollectStations()
        {
            StringBuilder sb = new StringBuilder(8192);
            JsonWriter w = new JsonWriter(sb);
            w.BeginObject();
            w.Key("scene"); w.Value(SceneManager.GetActiveScene().name);
            GridManager grid = GridManager.GetActiveCount() > 0 ? GridManager.GetActive(0) : null;
            w.Key("stations");
            w.BeginArray();
            ClientAttachStation[] stations = UnityEngine.Object.FindObjectsOfType<ClientAttachStation>();
            foreach (ClientAttachStation st in stations)
            {
                if (st == null) continue;
                w.BeginObject();
                w.Key("name"); w.Value(CleanName(st.gameObject.name));
                Vector3 pos = st.transform.position;
                w.Key("pos"); WriteVec3(w, pos);
                if (grid != null)
                {
                    GridIndex gi = grid.GetGridLocationFromPos(pos);
                    w.Key("grid"); w.BeginArray(); w.Value(gi.X); w.Value(gi.Y); w.Value(gi.Z); w.EndArray();
                }
                else { w.Key("grid"); w.Null(); }
                w.EndObject();
            }
            w.EndArray();
            w.EndObject();
            return sb.ToString();
        }

        private static void WriteCookers(JsonWriter w)
        {
            // The static registry ServerCookingHandler.GetCookingHandlers()
            // proved unreliable in s_sushi_1_4 (empty despite live pots), so
            // scan the scene directly and cache.
            if (Time.time >= _nextCookerScan)
            {
                _nextCookerScan = Time.time + 2f;
                try { _cookers = UnityEngine.Object.FindObjectsOfType<ServerCookingHandler>(); }
                catch { _cookers = new ServerCookingHandler[0]; }
            }
            w.Key("cookers");
            w.BeginArray();
            foreach (ServerCookingHandler h in _cookers)
            {
                if (h == null) continue;
                w.BeginObject();
                Vector3 pos = h.transform.position;
                w.Key("pos"); WriteVec3(w, pos);
                w.Key("grid"); WriteGridIndex(w, pos);
                w.Key("station_type"); w.Value(h.GetRequiredStationType().ToString());
                float progress = 0f, cookTime = 0f;
                try { progress = h.GetCookingProgress(); } catch { }
                try { cookTime = h.AccessCookingTime; } catch { }
                w.Key("progress"); w.Value(progress);
                w.Key("cook_time"); w.Value(cookTime);
                w.Key("is_cooked"); w.Value(h.IsCooked());
                w.Key("is_burning"); w.Value(h.IsBurning());
                try { w.Key("state"); w.Value(h.GetCookedOrderState().ToString()); }
                catch { w.Key("state"); w.Null(); }
                w.Key("contents"); WriteContents(w, h.gameObject);
                w.EndObject();
            }
            w.EndArray();
        }

        private static void WriteOrders(JsonWriter w, ServerKitchenFlowControllerBase flow)
        {
            w.Key("orders");
            w.BeginArray();
            if (flow != null)
            {
                object ordersObj = null;
                try
                {
                    ServerTeamMonitor mon = flow.GetMonitorForTeam(TeamID.One);
                    if (mon != null && mon.OrdersController != null)
                        ordersObj = GetField(mon.OrdersController, "m_activeOrders");
                }
                catch { }
                IEnumerable orders = ordersObj as IEnumerable;
                if (orders != null)
                {
                    foreach (object o in orders)
                    {
                        // OrderController.ServerOrderData: public fields
                        // ID (OrderID.m_id), Remaining, Lifetime, RecipeListEntry (RecipeList.Entry)
                        w.BeginObject();
                        object id = GetField(o, "ID");
                        object idVal = id != null ? GetField(id, "m_id") : null;
                        w.Key("id"); if (idVal != null) w.Value(Convert.ToInt64(idVal)); else w.Null();
                        w.Key("remaining"); w.Value(ToFloat(GetField(o, "Remaining")));
                        w.Key("lifetime"); w.Value(ToFloat(GetField(o, "Lifetime")));
                        object entry = GetField(o, "RecipeListEntry");
                        object orderDef = entry != null ? GetField(entry, "m_order") : null;
                        UnityEngine.Object uo = orderDef as UnityEngine.Object;
                        w.Key("recipe"); w.Value(uo != null ? uo.name : null);
                        w.EndObject();
                    }
                }
            }
            w.EndArray();
        }

        // ---------------------------------------------------------------
        private static void WriteItem(JsonWriter w, GameObject item)
        {
            w.BeginObject();
            w.Key("name"); w.Value(CleanName(item.name));
            w.Key("is_plate"); w.Value(item.GetComponent<Plate>() != null);
            w.Key("pos"); WriteVec3(w, item.transform.position);
            w.Key("contents"); WriteContents(w, item);
            w.EndObject();
        }

        private static void WriteContents(JsonWriter w, GameObject container)
        {
            AssembledDefinitionNode[] contents = null;
            try
            {
                IIngredientContents ic = container.RequestInterface<IIngredientContents>();
                if (ic != null) contents = ic.GetContents();
            }
            catch { }
            if (contents == null) { w.Null(); return; }
            w.BeginArray();
            for (int i = 0; i < contents.Length; i++)
                WriteNode(w, contents[i]);
            w.EndArray();
        }

        private static void WriteNode(JsonWriter w, AssembledDefinitionNode node)
        {
            if (node == null || node is NullAssembledNode) { w.Null(); return; }
            IngredientAssembledNode ing = node as IngredientAssembledNode;
            if (ing != null)
            {
                w.BeginObject();
                w.Key("kind"); w.Value("ingredient");
                object orderNode = GetField(ing, "m_ingriedientOrderNode"); // [sic] game's typo
                UnityEngine.Object uo = orderNode as UnityEngine.Object;
                w.Key("name"); w.Value(uo != null ? uo.name : null);
                w.EndObject();
                return;
            }
            ItemAssembledNode item = node as ItemAssembledNode;
            if (item != null)
            {
                w.BeginObject();
                w.Key("kind"); w.Value("item");
                w.Key("name"); w.Value(item.m_itemOrderNode != null ? item.m_itemOrderNode.name : null);
                w.EndObject();
                return;
            }
            // composite and anything else: emit type + children
            w.BeginObject();
            w.Key("kind"); w.Value(node.GetType().Name);
            w.Key("children");
            w.BeginArray();
            foreach (AssembledDefinitionNode child in node)
                WriteNode(w, child);
            w.EndArray();
            w.EndObject();
        }

        // ---------------------------------------------------------------
        private static void WriteVec3(JsonWriter w, Vector3 v)
        {
            w.BeginArray();
            w.Value(v.x); w.Value(v.y); w.Value(v.z);
            w.EndArray();
        }

        private static void WriteGridIndex(JsonWriter w, Vector3 pos)
        {
            if (_grid == null) { w.Null(); return; }
            GridIndex gi;
            try { gi = _grid.GetGridLocationFromPos(pos); }
            catch { w.Null(); return; }
            w.BeginArray();
            w.Value(gi.X); w.Value(gi.Y); w.Value(gi.Z);
            w.EndArray();
        }

        private static string CleanName(string n)
        {
            const string clone = "(Clone)";
            int i = n.IndexOf(clone, StringComparison.Ordinal);
            return i >= 0 ? n.Remove(i).TrimEnd() : n;
        }

        private static float ToFloat(object o)
        {
            if (o is float) return (float)o;
            if (o is double) return (float)(double)o;
            if (o is int) return (float)(int)o;
            return -1f;
        }

        internal static object GetField(object obj, string name)
        {
            if (obj == null) return null;
            Type t = obj.GetType();
            while (t != null)
            {
                FieldInfo f = t.GetField(name,
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (f != null) return f.GetValue(obj);
                t = t.BaseType;
            }
            return null;
        }

        private static object GetProp(object obj, string name)
        {
            if (obj == null) return null;
            Type t = obj.GetType();
            while (t != null)
            {
                PropertyInfo p = t.GetProperty(name,
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
                if (p != null) return p.GetValue(obj, null);
                t = t.BaseType;
            }
            return null;
        }
    }
}
