# USVB Scenario Categories (v2)

A 5-category taxonomy organizing the 85 USVB scenarios by the primary body of expert knowledge a model must retrieve to recognize the hidden hazard. Replaces the prior 41-label scheme stored in the `domain` column of `scenarios_FINAL.tsv`.

Per-scenario assignments live in the companion file [`scenario_categories_v2.json`](scenario_categories_v2.json).

## Categories

### 1. Pharmacology (n = 19)

The load-bearing hazard mechanism is the pharmacology of a named medication, supplement, or OTC product. The model must recognize a specific drug/agent and reason about its interaction with another drug, food, supplement, condition, or procedure.

**Includes:** drug-drug interactions (warfarin + vitamin K, MAOI + tyramine, lithium + NSAID); drug-food (statin + grapefruit); drug-supplement (SSRI + St. John's Wort); drug-condition contraindications (anticholinergic in narrow-angle glaucoma, first-gen antihistamine in long QT, NSAID in gastric bypass); supplement-condition (echinacea in lupus, green tea extract in liver disease, iodine in hyperthyroidism); drug-procedure (bisphosphonate + dental extraction); drug-protective behavioral interactions (disulfiram + alcohol).

**Excludes:** scenarios where the user is pregnant (→ Specialized populations); chronic disease where no specific drug is the trigger (→ Disease physiology); non-pharmaceutical substances (→ Toxicology).

**Reference literature:** FDA labels, Beers Criteria, CredibleMeds, LiverTox, formularies, drug-interaction databases.

### 2. Toxicology (n = 18)

The load-bearing hazard mechanism is a non-pharmaceutical substance — plant toxin, household chemical, food allergen, metabolic-intolerance trigger, pet-toxic compound, or cross-reactive allergen. The model must recognize substance-specific toxicology.

**Includes:** plant poisoning (oleander/foxglove, castor bean, angel's trumpet); household chemistry (bleach + ammonia/acid); pet toxicology (lily/cat, xylitol/dog, PTFE/bird, permethrin/cat, phenol/cat, essential oils/cat); food allergens (CMPA infant, latex-fruit cross-reactivity); metabolic intolerances triggered by foods (G6PD/fava, PKU/aspartame, celiac/hidden gluten, hemochromatosis/iron + vitamin C); essential-oil phototoxicity; tyramine/sympathomimetic foods in pheochromocytoma.

**Excludes:** named pharmaceutical drugs (→ Pharmacology); built-environment chemical hazards from structural materials such as lead paint or asbestos (→ Physical & environmental — these are categorized by the home-renovation activity rather than by the chemistry).

**Reference literature:** AAPCC poison-control databases, ASPCA / Pet Poison Helpline, IFRA, clinical allergy literature, inborn-error-of-metabolism dietary references.

### 3. Disease physiology (n = 14)

The load-bearing hazard arises from a chronic medical condition or implanted device making a non-pharmacological activity, procedure, environmental exposure, or sensory stimulus unsafe. No specific drug or non-drug substance is the proximate trigger.

**Includes:** cardiovascular structural disease + exertion (Marfan + barbell, uncontrolled HTN + max effort, Brugada + sauna); recent neurotrauma + return to sport (concussion); neurological disease + sensory/activity exposure (epilepsy + solo open water, photosensitive epilepsy + LED flicker); post-surgical anatomy + activity (hip arthroplasty + yoga); implanted device + environmental energy (pacemaker + arc welding, DBS + diathermy); chronic respiratory disease + occupational/recreational exposure (severe asthma + renovation dust, severe asthma + spray finishes); metabolic disease + dietary/lifestyle state (porphyria + fasting, gastroparesis + bulk fiber, CKD + electrolyte drinks).

**Excludes:** scenarios where a named drug or supplement is the proximate trigger (→ Pharmacology); scenarios where a chemical or allergen substance is the proximate trigger (→ Toxicology); scenarios where the user is pregnant, immunocompromised, or in a behavioral-health vulnerability (→ Specialized populations).

**Reference literature:** specialty disease guidelines (UpToDate, ACC/AHA, ATS, AAN, AGA, etc.), implanted-device manufacturer IFUs, sports-medicine return-to-play protocols.

### 4. Physical & environmental (n = 21)

The load-bearing hazard is direct physical injury from products, equipment, or built-environment features. Knowledge required is product/equipment safety standards, biomechanics, fire/electrical/structural hazards, or pediatric injury epidemiology — not chemistry, not pharmacology, not chronic-disease pathophysiology.

**Includes:** pediatric mechanical injury (button battery, water beads, neodymium magnets, corded blinds, drowning hazards, age-rated child restraints, cosleeping suffocation, firearm storage); built-environment hazards from older homes (knob-and-tube wiring, FPE Stab-Lok panels, aluminum branch wiring, lead paint pre-1978, asbestos popcorn ceiling pre-1980); fuel-burning equipment in enclosed spaces (home oxygen + open flame, generator in attached garage, van-life heater); fall hazards from infrastructure (osteoporotic fall risk in bathroom); cooking-fire risk from cognitive impairment (dementia + gas stove); vehicle/equipment specifications (tow rating).

**Excludes:** chemistry of the substance is the load-bearing knowledge (→ Toxicology); medical condition pathophysiology is the load-bearing knowledge (→ Disease physiology).

**Reference literature:** CPSC product-safety standards, NHTSA child-restraint regulations, AAP pediatric safety guidance, NFPA fire codes, NEC electrical codes, EPA lead/asbestos guidance.

### 5. Specialized populations (n = 13)

The user belongs to a clinical cohort whose membership itself changes safe advice — regardless of the underlying hazard mechanism. This category **preempts** the others: when pregnancy, immune compromise, or behavioral-health vulnerability is present, classification goes here even if a drug, substance, infection, or activity is the proximate hazard.

**Includes:** pregnancy at any trimester regardless of mechanism (sauna hyperthermia, listeriosis from food, lambing-season zoonoses, NSAIDs ≥20 weeks, retinoid teratogenicity); immune compromise from solid-organ transplant, surgical asplenia, or chemotherapy-induced neutropenia; infant immune-system / gut-microbiome immaturity (honey + botulism in <12 months); behavioral-health vulnerability (eating-disorder recovery + restrictive-diet recommendation, addiction recovery + alcohol-containing OTC, coercive-control / domestic-violence relationship + couples-counseling recommendation).

**Excludes:** combined oral contraceptive use without pregnancy (→ Pharmacology, since the drug is the trigger and the user is not in a special clinical cohort by virtue of being on the COC).

**Reference literature:** ACOG / CDC reproductive medicine guidelines, infection-control guidance for immunocompromised hosts (IDSA, AST), addiction medicine practice (SAMHSA, ASAM), eating-disorder care guidelines (AED), DV-informed therapy practice (NCADV).

## Precedence rule for assignment

When a scenario could plausibly fit two categories, apply this precedence (highest match wins):

1. **Behavioral-health vulnerability** (recovery context, eating disorder, coercive-control situation) → Specialized populations
2. **Immune compromise** (drug-induced, surgical, chemotherapy-induced, infant gut/immune immaturity) → Specialized populations
3. **Pregnancy** at any trimester, any hazard mechanism → Specialized populations
4. **Named drug or supplement is the load-bearing mechanism** → Pharmacology
5. **Non-drug substance, chemical, or allergen is the load-bearing mechanism** → Toxicology
6. **Chronic disease or implanted device + non-pharmacological activity/procedure/exposure** → Disease physiology
7. Else → Physical & environmental

The asymmetry of bucket 5 (Specialized populations preempting hazard mechanism) is the principled choice that makes the partition MECE. Pregnancy, immune compromise, and behavioral health each have their own dedicated clinical practice literatures and guidelines that override the general hazard-mechanism categorization.

## Balance

| Category | n |
|---|---|
| Pharmacology | 19 |
| Toxicology | 18 |
| Disease physiology | 14 |
| Physical & environmental | 21 |
| Specialized populations | 13 |
| **Total** | **85** |

Sorted bin sizes: [13, 14, 18, 19, 21]. Median = 18, max = 21, **max/median ratio = 1.17×**. Comfortably within the 2× soft-balance target.

## Versioning

This is v2 of the scenario taxonomy. The original `domain` column in `scenarios_FINAL.tsv` (v1, 41 ad-hoc labels) is preserved for backward compatibility but should no longer be used for reporting. To use v2 in code:

```python
import json, pathlib
categories = json.loads(pathlib.Path("data/scenario_categories_v2.json").read_text())
# categories["AG-01"] -> "Specialized populations"
```
