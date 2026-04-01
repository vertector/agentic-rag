from shared.schemas import PipelineSettings

def test_settings_split():
    print("Testing PipelineSettings with split kwargs...")
    settings = PipelineSettings(
        layout_threshold=0.8,
        max_new_tokens=256,
        vl_rec_backend="local",
        temperature=0.7
    )
    
    init_kwargs = settings.to_init_kwargs()
    predict_kwargs = settings.to_predict_kwargs()
    
    print(f"Init kwargs keys: {sorted(init_kwargs.keys())}")
    print(f"Predict kwargs keys: {sorted(predict_kwargs.keys())}")
    
    # Assertions for init_kwargs
    assert init_kwargs["vl_rec_backend"] == "native"
    assert "vl_rec_api_model_name" in init_kwargs
    assert "layout_threshold" in init_kwargs
    assert "temperature" not in init_kwargs  # temperature is predict-only
    assert "max_new_tokens" not in init_kwargs # max_new_tokens is predict-only
    print("PASS: init_kwargs correctly partitioned")
    
    # Assertions for predict_kwargs
    assert "temperature" in predict_kwargs
    assert predict_kwargs["temperature"] == 0.7
    assert "vl_rec_backend" not in predict_kwargs # backend is init-only
    assert "layout_threshold" in predict_kwargs
    assert predict_kwargs["layout_threshold"] == 0.8
    print("PASS: predict_kwargs correctly partitioned")

if __name__ == "__main__":
    try:
        test_settings_split()
        print("\nALL SCHEMA SPLIT TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
