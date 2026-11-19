# AI Usage Documentation – Snow-Proof Machine Learning

## 1. AI Tool Used

**Tool:** OpenAI ChatGPT (2026)  
**Website:** https://chatgpt.com/

Generative AI was extensively used throughout the development of the Snow-Proof machine learning component.

Most of the Python implementation, experimental scripts, testing commands and debugging instructions were generated using ChatGPT.

## 2. Scope of AI Usage

ChatGPT was used for:

- Generating Python code for data preparation and machine learning.
- Developing and adjusting Random Forest models.
- Creating scripts and terminal commands for testing and debugging.
- Supporting the creation and verification of historical training labels.
- Designing historical evaluation procedures and comparisons with simpler prediction methods.
- Analysing errors, interpreting test results and suggesting improvements.
- Providing guidance on Git branches and integrating the ML component into the team project.

The AI assistance was not limited to individual code suggestions. It covered substantial parts of the technical implementation.

## 3. Development Process

The machine learning component was developed through an iterative AI-assisted process rather than through a single prompt.

ChatGPT generated code and terminal commands, which were executed locally. Outputs, errors and test results were then shared with ChatGPT for analysis and further improvements.

This process included:

1. Preparing historical snow-depth data and training labels.
2. Developing and testing machine learning approaches.
3. Checking data quality and correcting implementation errors.
4. Evaluating predictions on historical years not included in the respective training periods.
5. Comparing Random Forest predictions with a historical baseline.
6. Updating the dataset and preparing the machine learning component for integration.

ChatGPT generated most of the code and testing procedures. Human involvement focused on providing project requirements, executing the suggested commands, reporting results, discussing alternatives and reviewing the model's limitations.

## 4. Critical Evaluation

AI-generated code and model predictions were not treated as automatically correct.

The development included repeated checks of data quality, model behaviour and prediction performance.

One important finding was that the Random Forest did not consistently outperform a simpler historical baseline. This limitation was explicitly examined rather than presenting the machine learning model as universally superior.

Successful technical tests also do not establish that the model's probabilities are perfectly calibrated or that its predictions represent actual future weather forecasts.

## 5. Historical Development Files

This branch contains experimental and intermediate versions of the machine learning implementation.

The Python scripts created for ML experiments, testing, debugging and evaluation were predominantly generated with ChatGPT.

This declaration covers the AI-assisted development work represented in this branch. Individual files may also contain code originating from the original team project or other contributors; this declaration does not attribute such third-party contributions to ChatGPT.

Some experiments were subsequently revised or replaced. The final implementation was integrated into the team's `skeleton-one-page` branch.

The historical files are retained to document the development process and do not necessarily represent the current production implementation.

## 6. Limitations of AI Assistance

Although generative AI made it possible to implement and test a relatively complex ML component within a limited timeframe, it introduced several challenges:

- AI-generated code required execution and verification.
- Some suggested approaches needed corrections or further testing.
- Model performance had to be assessed independently of the AI's explanations.
- Extensive AI assistance made understanding and maintaining the generated code more challenging.

The responsibility for accurately declaring AI usage, verifying submitted results and communicating the model's limitations remains with the project team.

## 7. Reference

OpenAI. (2026). *ChatGPT* [Large language model]. https://chatgpt.com/

---

