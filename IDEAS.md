Feature Engineering Ideas

* Error column: CountOfError12Events
* Telemetry: Interactions Between the four variables? something to capture the 24h signal?
* Where are my machine metadata features?
* Visualize nulls
* Add records to recapture "scheduled maintenance with zero repairs"

Open Question

* We only have 7000 records when the target variable is actually known. But this means we cannot test "how early in advance do we know a failure will occur". If only there was a way to "estimate next failure date"?
* Why are there so many errors at the 6am sample?
* Would it provide BV to predict which component failed (not just that a failure occurred)? maybe save technician time by letting them know which gear to bring.
* What to do about unscheduled maintenance? Build a classifier around those? What's different about those events. 

Class imbalance: 

* Class weight parameter to lightgbm. 
* Try SMOTE
* Choose different decision thresholds (depends on Cost(FN) vs Cost(FP))

Model Selection

* Sequence models (e.g., RNN) given the previous weeks' worth of telemetry
* A simple VAE generates maybe 32-dimensional embedding on every 24h window. Train xgboost on CONCAT(those embeddings + handcrafted features)
* Switch to sklearn pipelines for improved expressivity. 

Cool Extensions

* A viz showcasing, for a given prediction, which variables drive it (using SHAP features in h2o).
* A viz showcasing, for a given prediction, the timeseries context + and time-agg features summarizing it