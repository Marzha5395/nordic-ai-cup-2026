"""Refresh the runtime proxy token/url of local colab-cli sessions from the
server-side assignment listing (the CLI never refreshes them; they expire ~1h)."""
import json
import base64
import time
from colab_cli.common import state
sessions, assignments = state.sync_sessions()
by_ep = {a.endpoint: a for a in assignments}
path = state.store.path if hasattr(state.store, "path") else None
for name, s in sessions.items():
    a = by_ep.get(s.endpoint)
    if a is None:
        print(f"{name}: no server assignment")
        continue
    info = a.runtime_proxy_info
    s.token = info.token
    s.url = info.url
    state.store.add(s)
    payload = json.loads(base64.urlsafe_b64decode(info.token.split(".")[1] + "==="))
    print(f"{name}: token refreshed, expires in {(payload['exp'] - time.time())/60:.0f} min, url={info.url}")
