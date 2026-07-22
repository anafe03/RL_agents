# ALS World — Live Bridge (Option B)

Play the **actual sound of your Ableton project** and have the world react to
it live. Ableton renders the audio (all your real instruments/plugins); this
bridge streams its transport + per-stem levels to the browser world.

```
Ableton Live ──OSC/UDP──▶ bridge.py ──WebSocket──▶ alsworld/index.html
```

## One-time setup

1. **Install AbletonOSC** (free): download from
   https://github.com/ideoforms/AbletonOSC and drop the `AbletonOSC` folder
   into Ableton's *Remote Scripts* / *MIDI Remote Scripts* folder.
2. In Ableton → **Settings → Link/Tempo/MIDI → Control Surface**, choose
   **AbletonOSC**. You should see it confirm in Ableton's status bar.

## Run it

```bash
cd alsworld/bridge
pip install -r requirements.txt      # or: uv pip install -r requirements.txt
python bridge.py
```

You should see `listening for Ableton …`. Then:

3. Open `alsworld/index.html`, drop your **.als** (builds the world),
   and click **connect live**.
4. Press **Play** in Ableton. The wanderer follows Ableton's real playhead
   and each stem's region reacts to that track's live level.

## Notes

- The `.als` gives the **world's shape** (arrangement → terrain, stems →
  regions, locators → gateways). Ableton gives the **sound + live motion**.
- Ports default to AbletonOSC's out-of-the-box `11000` (send) / `11001`
  (reply); WebSocket is `8765`. Override with `--send-port`, `--recv-port`,
  `--ws-port` if you changed them.
- No Ableton running? The browser has a **Sim** mode so you can still see the
  world breathe with fake data.
