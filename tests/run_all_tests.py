"""Run all MNN tests."""
import sys; sys.path.insert(0,"..")
modules=[("Core","test_core"),("Geometry","test_geometry"),("Algebra","test_algebra"),("Chaos","test_chaos")]
passed=failed=0
for name,mod in modules:
    print(f"\n{'='*45}\n  {name}\n{'='*45}")
    try:
        m=__import__(mod); fns=[v for k,v in vars(m).items() if k.startswith("test_") and callable(v)]
        for fn in fns:
            try: fn(); print(f"  ✓ {fn.__name__}"); passed+=1
            except Exception as e: print(f"  ✗ {fn.__name__}: {e}"); failed+=1
    except Exception as e: print(f"  Import error: {e}"); failed+=1
print(f"\n{'='*45}\n  RESULTS: {passed} passed, {failed} failed\n{'='*45}")
