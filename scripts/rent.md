# Renting a box for phase 2

## What you need first

1. A vast.ai account with credit: https://cloud.vast.ai/
2. An API key: https://cloud.vast.ai/account/
3. Your SSH public key registered on the account (`~/.ssh/id_ed25519.pub`)

## Sizing

The base model is Qwen3-1.7B in bf16 (~3.4 GB) plus two KV caches. The policy
head is 297K parameters. This is a small job — bandwidth-bound, not
capacity-bound, so pick on memory bandwidth rather than VRAM.

| GPU | VRAM | Bandwidth | Fit |
|---|---|---|---|
| RTX 4090 | 24 GB | ~1.0 TB/s | plenty, cheapest sensible option |
| A100 40GB | 40 GB | ~1.5 TB/s | comfortable, faster |
| H100 | 80 GB | ~3.3 TB/s | overkill for 1.7B; right for the phase 5 27B run |

A 4090 is ~8× the memory bandwidth of the M4, so hours become minutes.

## Steps

```bash
pip install vastai
vastai set api-key <YOUR_KEY>

# find a box: 24GB+, good bandwidth, cheap, reliable
vastai search offers 'gpu_ram>=24 num_gpus=1 rentable=true dph<0.5' \
  --order 'dph+' --limit 20

vastai create instance <OFFER_ID> \
  --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime \
  --disk 40 --ssh --direct

vastai show instances          # wait for 'running', note ssh host/port
```

Then from this repo:

```bash
rsync -avz --exclude results --exclude .git --exclude '*_haystack.txt' \
  -e "ssh -p <PORT>" . root@<HOST>:~/kvAdmission/

ssh -p <PORT> root@<HOST> 'bash ~/kvAdmission/scripts/provision.sh'

rsync -avz -e "ssh -p <PORT>" \
  root@<HOST>:~/kvAdmission/results/ ./results-remote/
```

## Destroy it when done

```bash
vastai destroy instance <INSTANCE_ID>
```

**Billing runs until the instance is destroyed, not until the job finishes.**
Stopping is not destroying. Check `vastai show instances` before you walk away.

## Note

vast.ai hosts are other people's machines. Upload research code and public
model weights; never credentials.
