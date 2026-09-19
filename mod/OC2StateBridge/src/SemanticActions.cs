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

            IClientHandlePlacement ihp = objs.m_iHandlePlacement;
            if (ihp != null)
            {
                Send("Place", pc.gameObject, ihp as MonoBehaviour);
                return "place:" + ((ihp as MonoBehaviour) != null ? (ihp as MonoBehaviour).name : "?");
            }
            Send("Take", pc.gameObject, (GameObject)null);  // drop
            return "drop";
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
