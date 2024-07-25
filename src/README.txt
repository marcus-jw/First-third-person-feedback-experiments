Experiment 1: Log probs on datasets using LLM
- generate_pov_themes_data: generate datasets
- eval_llm: evaluate LLM feedback from different points of view (pov) on the pov datasets


Experiment 2: Evaluate preference models
- generate_pov_themes_data: reuse the datasets from experiment 1
- generate_hh_labels: generate labels from different points of view
- train_hh_pm: train a preference model (PM) on the hh dataset with labels from above
- eval_pm: evaluate the preference model


For running on new data when we train and test on half:
1. Get labels: eval_llm/eval_gpt.py
2. Run train_hh_pm/prep_train_on_half_of_personalization.py to generate combined data
3. Run train_hh_pm/scripy_train_pm.bash to train model
4. Run eval_pm/eval.py to run model on data
5. Run eval_pm/plot_results_averagePMs.py to make plots
6. Inspect plot in data/plots/pm_trainonperso_grm_1epochs.png

for num_perspective in {1..3..2}; do for num_run in {1..3}; do echo /home/constantinweisser/anaconda3/envs/mats/bin/python /nas/ucb/constantinweisser/First-third-person-feedback-experiments/src/eval_pm/eval.py $num_perspectiv
e $num_run; done; done

