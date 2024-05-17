source pyorbbecsdk
```
cd camera/orbbec/pyorbbecsdk
export PYTHONPATH=$PYTHONPATH:$(pwd)/install/lib/ # DON'T forget do this
```

1. collect_demo_data.py - collecting raw data/poses. saves it in data/demonstrations folder
2. prepare_demo_data.ipynb - prepare data for taxpose training. save it to data/train_data
3. train_residual_flow.py - models/taxpose/scripts/README.md to train. 
    WANDB account for analytics
4. test_inference.py - command similar to 3. Config files in models/taxpose/configs/commands/mfi/waterproof