import xgboost as xgb

def train_model(X_train, y_train, X_train_scaled=None, X_test=None, y_test=None, X_test_scaled=None):
    """Train an XGBoost regression model."""
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test the model
    if X_test is not None and y_test is not None:
        test_loss = model.score(X_test, y_test)
        print(f"Test loss: {test_loss/len(X_test)}")

    return model