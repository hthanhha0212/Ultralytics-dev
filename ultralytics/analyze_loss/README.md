# Ultralytics Loss Analysis (GPU)

This directory hosts a stand-alone CLI that replays a YOLO detection dataset with a trained checkpoint and reports the
images that currently incur the largest training loss. The helper uses the existing Ultralytics modules but **does not
modify** any internal library files.

## Running

```powershell
python ultralytics/analyze_loss/loss_cli.py `
    --model runs/detect/train/weights/best.pt `
    --data ultralytics/cfg/datasets/coco8.yaml `
    --split train `
    --batch 4 `
    --workers 4 `
    --topk 50 `
    --save-csv runs/loss/train_loss_rank.csv `
    --half
```

If you do not have a YAML, pass `--images path\to\dataset\images\train` (expects `labels/` in the usual format).

### Key Options

| Flag            | Description                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------- |
| `--model`       | Path to the trained YOLO `.pt` checkpoint (required).                                       |
| `--data`        | Dataset YAML path.                                                                          |
| `--images`      | Alternative to `--data` when only an image directory is available.                          |
| `--split`       | Dataset split key (`train`, `val`, `test`, etc.).                                           |
| `--batch`       | Batch size when iterating the dataset.                                                      |
| `--workers`     | Dataloader workers.                                                                         |
| `--topk`        | Number of highest-loss images to log.                                                       |
| `--max-samples` | Optional cap on the number of images processed.                                             |
| `--half`        | Run the scoring pass in FP16 when CUDA is available.                                        |
| `--save-csv`    | Path to dump the full ranking as `image,total_loss,box_loss,cls_loss,dfl_loss`.            |
| `--device`      | Explicit torch device string. Defaults to CUDA if available, otherwise CPU.                 |

The CLI automatically recreates the loss criterion and merges your checkpoint’s stored hyperparameters with the
defaults to avoid dependency on how the checkpoint was saved. It then logs the top contributors along with their
box/class/DFL components so you can inspect hard examples or feed the CSV into downstream QA tooling.
