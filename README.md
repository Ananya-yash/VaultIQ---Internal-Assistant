---
title: Vaultiq Api
emoji: 🐢
colorFrom: purple
colorTo: gray
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
To renew qdrant database:
python -c "
from core.qdrant_client import get_qdrant_client
client = get_qdrant_client()
client.delete_collection('company_docs')
print('Collection deleted.')
"
python -m core.ingestion