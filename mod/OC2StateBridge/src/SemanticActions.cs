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
            // RICE: pots cook rice; nori/chopped fillings go on plates.
            // PLATE: ServerPlacementContainer.CanHandlePlacement reverse-
            // combines — the pot POURS its cooked contents into the held
            // plate (the scoop!). The scan often finds no placement handler
            // for a plate (plate isn't placeable ON the pot), so INTERACT
            // dropped the plate on the floor and plated could never be
            // trained (probe run36: 3x interact, plate ended on the floor).
            bool isCookable = held.name.Contains("Rice") && !held.name.StartsWith("Chopped");
            bool isPlate = held.GetComponent<Plate>() != null;
            if (isCookable || isPlate)
            {
                ClientPlacementContainer pot = FindNearestPotContainer(pc.transform.position, 3.0f);
                if (pot != null)
                {
                    Send("Place", pc.gameObject, pot);
                    return "place-into:" + pot.name;
                }
            }
            // Plate-lock: nori/chopped fillings belong ON a plate. Same
            // direct-event pattern as the pot-lock — the scan's placement
            // referral near cluttered counters is unreliable, and assembly
            // (deepest plate observed: rice only) cannot be trained without
            // a deterministic add-to-plate path.
            bool isPlateFood = held.name == "Seaweed" || held.name.StartsWith("Chopped");
            if (isPlateFood)
            {
                ClientPlacementContainer plate = FindNearestPlateContainer(pc.transform.position, 3.0f);
                if (plate != null)
                {
                    Send("Place", pc.gameObject, plate);
                    return "place-onto:" + plate.name;
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

        private static ClientPlacementContainer[] _containerCache;
        private static float _containerCacheTime;

        /// <summary>All ClientPlacementContainer in the scene, rescanned at
        /// most every 2s. Positions are re-read per query (plates move).</summary>
        private static ClientPlacementContainer[] ContainerCache()
        {
            if (_containerCache == null || Time.time > _containerCacheTime)
            {
                _containerCache = UnityEngine.Object.FindObjectsOfType<ClientPlacementContainer>();
                _containerCacheTime = Time.time + 2f;
            }
            return _containerCache;
        }

        /// <summary>Nearest pot's placement container within maxDist of pos,
        /// or null.</summary>
        private static ClientPlacementContainer FindNearestPotContainer(Vector3 pos, float maxDist)
        {
            ClientPlacementContainer best = null;
            float bestD = maxDist;
            foreach (ClientPlacementContainer c in ContainerCache())
            {
                if (c == null || !c.name.Contains("pot")) continue;
                float d = Vector3.Distance(pos, c.transform.position);
                if (d < bestD) { bestD = d; best = c; }
            }
            return best;
        }

        /// <summary>Nearest PLATE's placement container within maxDist of
        /// pos, or null. Plate containers carry a Plate component in their
        /// hierarchy.</summary>
        private static ClientPlacementContainer FindNearestPlateContainer(Vector3 pos, float maxDist)
        {
            ClientPlacementContainer best = null;
            float bestD = maxDist;
            foreach (ClientPlacementContainer c in ContainerCache())
            {
                if (c == null) continue;
                if (c.GetComponentInParent<Plate>() == null) continue;
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

        /// <summary>Dispense from the crate nearest to (x,z), bypassing the
        /// proximity scan entirely (same direct-event pattern as the pot
        /// lock). The scan keeps losing to loose items near the cucumber
        /// crate (fire extinguisher at (14.4,-2.4), floor litter), making
        /// scripted fetches unreliable; the chef must still stand close,
        /// but facing/scan-cone no longer matter.</summary>
        public static string FetchCrate(int player, float x, float z, ManualLogSource log)
        {
            PlayerControls pc = GetPC(player);
            if (pc == null) return "no-player";
            IPlayerCarrier carrier = (IPlayerCarrier)pc.GetComponent(typeof(IPlayerCarrier));
            GameObject held = carrier != null ? carrier.InspectCarriedItem(PlayerAttachTarget.Default) : null;
            if (held != null) return "hands-full:" + held.name;
            ClientPickupItemSpawner best = null;
            float bestD = 3.0f;
            foreach (ClientPickupItemSpawner s in UnityEngine.Object.FindObjectsOfType<ClientPickupItemSpawner>())
            {
                if (s == null || !s.name.StartsWith("DispenserCrate")) continue;
                Vector3 d = s.transform.position - new Vector3(x, 0f, z);
                d.y = 0f;
                if (d.magnitude < bestD) { bestD = d.magnitude; best = s; }
            }
            if (best == null) return "no-crate";
            Send("PickUp", pc.gameObject, best);
            return "fetch:" + best.name;
        }

        /// <summary>Pick up the loose plate nearest to the player, via the
        /// same PickUp event the proximity scan sends (GameObject overload —
        /// m_TheOriginalHandlePickup is a GameObject). Plates are loose
        /// items in this level; the "plate station" is the delivery hatch,
        /// NOT a dispenser (its CanAddItem requires a ClientPlate).</summary>
        public static string FetchPlate(int player, ManualLogSource log)
        {
            PlayerControls pc = GetPC(player);
            if (pc == null) return "no-player";
            IPlayerCarrier carrier = (IPlayerCarrier)pc.GetComponent(typeof(IPlayerCarrier));
            GameObject held = carrier != null ? carrier.InspectCarriedItem(PlayerAttachTarget.Default) : null;
            if (held != null) return "hands-full:" + held.name;
            PhysicalAttachment best = null;
            float bestD = 3.0f;
            foreach (PhysicalAttachment pa in UnityEngine.Object.FindObjectsOfType<PhysicalAttachment>())
            {
                if (pa == null || pa.gameObject.GetComponent<Plate>() == null) continue;
                Vector3 d = pa.transform.position - pc.transform.position;
                d.y = 0f;
                if (d.magnitude < bestD) { bestD = d.magnitude; best = pa; }
            }
            if (best == null) return "no-plate";
            Send("PickUp", pc.gameObject, best.gameObject);
            return "fetch-plate:" + best.name;
        }

        /// <summary>Place the held item onto the chopping board nearest to
        /// (x,z) via its ClientAttachStation directly (no proximity scan).
        /// </summary>
        public static string PlaceOnBoard(int player, float x, float z, ManualLogSource log)
        {
            PlayerControls pc = GetPC(player);
            if (pc == null) return "no-player";
            IPlayerCarrier carrier = (IPlayerCarrier)pc.GetComponent(typeof(IPlayerCarrier));
            GameObject held = carrier != null ? carrier.InspectCarriedItem(PlayerAttachTarget.Default) : null;
            if (held == null) return "empty-handed";
            ClientAttachStation best = null;
            float bestD = 3.0f;
            foreach (ClientAttachStation s in UnityEngine.Object.FindObjectsOfType<ClientAttachStation>())
            {
                if (s == null || !s.name.Contains("chopping_board")) continue;
                Vector3 d = s.transform.position - new Vector3(x, 0f, z);
                d.y = 0f;
                if (d.magnitude < bestD) { bestD = d.magnitude; best = s; }
            }
            if (best == null) return "no-board";
            Send("Place", pc.gameObject, best);
            return "place-board:" + best.name;
        }
    }
}
