"""Run all MNN tests — Phase 1 + Phase 2."""
import sys; sys.path.insert(0,"..")

modules = [
    ("Core Math Engine",      "test_core"),
    ("Geometry & Topology",   "test_geometry"),
    ("Algebra & Groups",      "test_algebra"),
    ("Chaos Theory",          "test_chaos"),
    ("Phase 2 — Advanced",    "test_advanced"),
]

passed = failed = 0
for name, mod in modules:
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    try:
        m    = __import__(mod)
        fns  = [v for k,v in vars(m).items() if k.startswith("test_") and callable(v)]
        for fn in fns:
            try:
                fn()
                print(f"  ✓ {fn.__name__}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {fn.__name__}: {e}")
                failed += 1
    except Exception as e:
        print(f"  Import error: {e}")
        failed += 1

print(f"\n{'='*50}")
print(f"  RESULTS: {passed} passed, {failed} failed")
print(f"{'='*50}")

# Phase 3 is added below via patching
_modules_ext = [("Phase 3 — Vector Calculus", "test_vector_calculus")]
for name, mod in _modules_ext:
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    try:
        m   = __import__(mod)
        fns = [v for k,v in vars(m).items() if k.startswith("test_") and callable(v)]
        for fn in fns:
            try: fn(); print(f"  ✓ {fn.__name__}"); passed+=1
            except Exception as e: print(f"  ✗ {fn.__name__}: {e}"); failed+=1
    except Exception as e: print(f"  Import error: {e}"); failed+=1

print(f"\n{'='*50}")
print(f"  GRAND TOTAL: {passed} passed, {failed} failed")
print(f"{'='*50}")

# Phase 4 manifold learning tests
for name, mod in [("Phase 4 — Manifold Learning", "test_manifold_learning")]:
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    try:
        m   = __import__(mod)
        fns = [v for k,v in vars(m).items() if k.startswith("test_") and callable(v)]
        for fn in fns:
            try: fn(); print(f"  ✓ {fn.__name__}"); passed+=1
            except Exception as e: print(f"  ✗ {fn.__name__}: {e}"); failed+=1
    except Exception as e: print(f"  Import error: {e}"); failed+=1

print(f"\n{'='*50}")
print(f"  GRAND TOTAL: {passed} passed, {failed} failed")
print(f"{'='*50}")

for name, mod in [("Phase 5 — Chaos Simulation", "test_chaos_simulation")]:
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    try:
        m   = __import__(mod)
        fns = [v for k,v in vars(m).items() if k.startswith("test_") and callable(v)]
        for fn in fns:
            try: fn(); print(f"  ✓ {fn.__name__}"); passed+=1
            except Exception as e: print(f"  ✗ {fn.__name__}: {e}"); failed+=1
    except Exception as e: print(f"  Import error: {e}"); failed+=1

print(f"\n{'='*50}")
print(f"  GRAND TOTAL: {passed} passed, {failed} failed")
print(f"{'='*50}")
