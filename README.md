# Credit-Risk-traing-model-
credit risk traing model for the bank called rati bank 

#Credit Scoring Business Understanding
Basel II and the Need for Interpretability

*The Basel II Accord emphasizes accurate risk measurement, transparency, and regulatory accountability in credit risk models. This requires models that are not only predictive but also interpretable and well-documented, allowing financial institutions and regulators to understand why a customer is classified as high or low risk. 
An interpretable model supports auditability, stress testing, and governance requirements, ensuring that risk estimates used for capital allocation and pricing can be justified and explained.

#Proxy Variable for Default and Associated Risks

*In the absence of a direct “default” label, a proxy target variable is created to approximate default-like behavior (e.g., high claim frequency or extreme loss ratios). 
This is necessary to enable supervised learning and risk estimation. However, relying on a proxy introduces business risks, including label noise, misclassification of customers, and potential bias if the proxy does not fully represent true default behavior. Poorly defined proxies can lead to incorrect pricing, unfair customer treatment, and suboptimal capital allocation.

#Model Complexity vs Interpretability Trade-offs

*Simple models such as Logistic Regression with Weight of Evidence (WoE) offer high interpretability, stability, and ease of regulatory approval, making them well-suited for regulated financial environments. However, they may sacrifice predictive power. In contrast, complex models like Gradient Boosting can capture non-linear relationships and improve accuracy but reduce transparency and are harder to explain, validate, and govern. 
In a regulated context, the trade-off involves balancing predictive performance with model explainability, compliance, and trust, often favoring simpler models for core decision-making while using complex models as supplementary tools.