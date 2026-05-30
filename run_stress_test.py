import logging
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.production.squad_uplift_integration import SquadUpliftIntegration
from src.production.feature_registry import FeatureRegistry
from src.production.operational_validator import OperationalStressTester
from src.experimental.poisson_squad_uplift import PoissonSquadUplift

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("stress_test_runner")

def main():
    logger.info("=" * 60)
    logger.info("STRESS TEST WITH TYPE IMMUNITY FIX")
    logger.info("=" * 60)
    
    # Initialize components
    logger.info("Initializing components...")
    registry = FeatureRegistry()
    approved_features = registry.get_approved_features()
    logger.info(f"Approved features: {approved_features}")
    
    # Initialize uplift model with approved feature columns
    uplift_model = PoissonSquadUplift(feature_columns=approved_features)
    
    # Fit model with minimal training data for stress test
    logger.info("Fitting model with minimal training data...")
    train_data = pd.DataFrame({
        "continuity_index": [0.7, 0.8, 0.9, 0.6, 0.75],
        "defenders_continuity": [0.65, 0.75, 0.85, 0.55, 0.70],
        "lambda_base": [1.2, 1.5, 1.8, 1.0, 1.4],
        "target_goals": [1, 2, 2, 1, 1]
    })
    uplift_model.fit(train_data, "target_goals", "lambda_base")
    
    integration = SquadUpliftIntegration(
        uplift_model=uplift_model,
        registry=registry,
        shadow_mode=True
    )
    
    tester = OperationalStressTester(integration)
    
    # Run stress test with 100 iterations (as per user's audit)
    logger.info("Running 100-iteration stress test...")
    results = tester.run_stress_test(n_iterations=100)
    
    # Analyze results
    logger.info("=" * 60)
    logger.info("STRESS TEST RESULTS")
    logger.info("=" * 60)
    for status, count in results.items():
        logger.info(f"{status}: {count}")
    
    # Certification check
    logger.info("=" * 60)
    logger.info("CERTIFICATION CHECK")
    logger.info("=" * 60)
    
    failsafe_errors = results.get("FAILSAFE_ERROR", 0)
    if failsafe_errors == 0:
        logger.info("✓ TYPE IMMUNITY: PASS - No FAILSAFE_ERROR detected")
        logger.info("✓ SYSTEM READY FOR GO-LIVE")
    else:
        logger.error(f"✗ TYPE IMMUNITY: FAIL - {failsafe_errors} FAILSAFE_ERROR detected")
        logger.error("✗ SYSTEM NOT READY FOR GO-LIVE")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
