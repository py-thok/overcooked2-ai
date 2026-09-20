using System;
using System.Reflection;
using BepInEx.Logging;
using UnityEngine;

namespace OC2StateBridge
{
    /// <summary>
    /// Semantic actions that bypass the button/edge pipeline entirely and call
    /// the game's own interaction events (ClientMessenger.ChefEventMessage),
    /// using the game's own proximity scan (PlayerControls.CurrentInteractionObjects)
    /// to resolve targets — exactly what real input would trigger.
    ///
    /// Why: raw button replacement fights the event-claim ordering between
    /// multiple readers; driving the semantic events is race-free.
    /// </summary>
    public static class SemanticActions
    {
        private static MethodInfo _chefEventMB;   // (type, chef, MonoBehaviour)
        private static MethodInfo _chefEventGO;   // (type, chef, GameObject)
        private static Type _chefEventType;

        private static void EnsureInit()
        {
            if (_chefEventMB != null) return;
            Type messenger = typeof(GameUtils).Assembly.GetType("ClientMessenger");
            _chefEventType = typeof(GameUtils).Assembly.GetType(
                "Team17.Online.Multiplayer.Messaging.ChefEventMessage+ChefEventType");
            foreach (MethodInfo m in messenger.GetMethods(
                BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic))
            {
                if (m.Name != "ChefEventMessage") continue;
                ParameterInfo[] ps = m.GetParameters();
                if (ps.Length == 3 && ps[2].ParameterType == typeof(MonoBehaviour)) _chefEventMB = m;
                if (ps.Length == 3 && ps[2].ParameterType == typeof(GameObject)) _chefEventGO = m;
            }
        }

        private static void Send(string eventName, GameObject chef, object target)
        {
            EnsureInit();
            object type = Enum.Parse(_chefEventType, eventName);
            if (target is MonoBehaviour mb && _chefEventMB != null)
                _chefEventMB.Invoke(null, new object[] { type, chef, mb });
            else if (target is GameObject go && _chefEventGO != null)
                _chefEventGO.Invoke(null, new object[] { type, chef, go });
            else if (target == null && _chefEventGO != null)
                _chefEventGO.Invoke(null, new object[] { type, chef, null });
        }

        private static PlayerControls GetPC(int player)
        {
            GameObject[] players = GameObject.FindGameObjectsWithTag("Player");
            Array.Sort(players, delegate (GameObject a, GameObject b)
            {
                return string.CompareOrdinal(a.name, b.name);
            });
            if (player < 0 || player >= players.Length) return null;
            return players[player].GetComponent<PlayerControls>();
        }

        /// <summary>Pickup if empty-handed, place/drop if holding. Returns a
        /// short string describing what happened (for logging/debug).</summary>
        public static string Interact(int player, ManualLogSource log)
        {
            PlayerControls pc = GetPC(player);
            if (pc == null) return "no-player";
            pc.UpdateNearbyObjects();
            PlayerControls.InteractionObjects objs = pc.CurrentInteractionObjects;
            IPlayerCarrier carrier = (IPlayerCarrier)pc.GetComponent(typeof(IPlayerCarrier));
            GameObject held = carrier != null ? carrier.InspectCarriedItem(PlayerAttachTarget.Default) : null;

            if (held == null)
            {
                if (objs.m_iHandlePickup != null)
                {
                    Send("PickUp", pc.gameObject, objs.m_TheOriginalHandlePickup);
                    return "pickup:" + (objs.m_TheOriginalHandlePickup != null
                        ? objs.m_TheOriginalHandlePickup.name : "?");
                }
                if (objs.m_interactable != null && objs.m_interactable.UsePlacementButton)
                {
                    Send("TriggerInteract", pc.gameObject, objs.m_interactable);
                    return "trigger:" + objs.m_interactable.name;
                }
                return "nothing-nearby";
            }

            // When carrying an ingredient near a pot, place directly into the
            // pot's ClientPlacementContainer BEFORE consulting the proximity
            // scan: the scan resolves the cooker station whose referral logic
            // fails intermittently (~50%), leaving the chef holding the item.
            // Direct placement into the container is deterministic.
            if (!held.name.StartsWith("utensil_") && !held.name.StartsWith("equipment_"))
            {
                ClientPlacementContainer pot = FindNearestPotContainer(pc.transform.position, 3.0f);
                if (pot != null)
                {
                    Send("Place", pc.gameObject, pot);
                    return "place-into:" + pot.name;
                }
            }
            IClientHandlePlacement ihp = objs.m_iHandlePlacement;
            if (ihp != null)
            {
                Send("Place", pc.gameObject, ihp as MonoBehaviour);
                return "place:" + ((ihp as MonoBehaviour) != null ? (ihp as MonoBehaviour).name : "?");
            }
            Send("Take", pc.gameObject, (GameObject)null);  // drop
            return "drop";
        }

        private static ClientPlacementContainer[] _potCache;
        private static float _potCacheTime;

        /// <summary>Nearest pot's placement container within maxDist of pos,
        /// or null. Component scan cached for 2s.</summary>
        private static ClientPlacementContainer FindNearestPotContainer(Vector3 pos, float maxDist)
        {
            if (_potCache == null || Time.time > _potCacheTime)
            {
                ClientPlacementContainer[] all = UnityEngine.Object.FindObjectsOfType<ClientPlacementContainer>();
                System.Collections.Generic.List<ClientPlacementContainer> pots =
                    new System.Collections.Generic.List<ClientPlacementContainer>();
                foreach (ClientPlacementContainer c in all)
                {
                    if (c != null && c.name.Contains("pot")) pots.Add(c);
                }
                _potCache = pots.ToArray();
                _potCacheTime = Time.time + 2f;
            }
            ClientPlacementContainer best = null;
            float bestD = maxDist;
            foreach (ClientPlacementContainer c in _potCache)
            {
                if (c == null) continue;
                float d = Vector3.Distance(pos, c.transform.position);
                if (d < bestD) { bestD = d; best = c; }
            }
            return best;
        }

        /// <summary>Throw the held item (game's own throw event).</summary>
        public static string ThrowItem(int player, ManualLogSource log)
        {
            PlayerControls pc = GetPC(player);
            if (pc == null) return "no-player";
            IPlayerCarrier carrier = (IPlayerCarrier)pc.GetComponent(typeof(IPlayerCarrier));
            if (carrier == null || carrier.InspectCarriedItem(PlayerAttachTarget.Default) == null)
                return "empty-handed";
            Send("Throw", pc.gameObject, (GameObject)null);
            return "throw";
        }
    }
}
