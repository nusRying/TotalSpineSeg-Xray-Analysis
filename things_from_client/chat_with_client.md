Sunshine Bizz
8:08 PM
Hello!

I'd like to invite you to take a look at the job I've posted. Please submit a proposal if you're available and interested.

Sunshine B.

Umair Ejaz
8:08 PM
Hi,

Thank you for inviting me to this project — I’d be glad to help adapt and deliver an existing 3D ML framework for a 2D X-ray-focused medical imaging use case.

I have experience working with Python and PyTorch-based machine learning pipelines, including computer vision workflows, model adaptation, image preprocessing, and segmentation-related tasks. I am comfortable working with existing codebases, understanding their structure quickly, and modifying them carefully to fit new data modalities and project requirements.

Scope Understanding
⚪ Adapt an existing 3D framework for use with 2D X-ray images
⚪ Update the data pipeline and preprocessing for 2D medical imaging input
⚪ Modify model architecture or training logic where required
⚪ Support segmentation-focused development and validation
⚪ Maintain clean, reliable, and reproducible code for ongoing development

My Approach
⚪ Review the current framework, package structure, and intended medical use-case
⚪ Identify the components that need to be converted from 3D to 2D
⚪ Update data loaders, preprocessing, augmentation, and model input/output flow
⚪ Refine training and inference scripts to ensure compatibility and stability
⚪ Test, debug, and improve the adapted pipeline for practical use

Deliverables
⚪ Adapted 2D-compatible Python/PyTorch codebase
⚪ Updated training and inference pipeline for X-ray images
⚪ Necessary architecture or preprocessing modifications
⚪ Documentation/comments for key implementation changes
⚪ Support for debugging and further refinement during development

Milestones
⚪ Milestone 1 – Framework Review & Adaptation Plan
Review the existing 3D framework, understand dependencies, and define the required conversion path.

⚪ Milestone 2 – 2D Pipeline & Model Adaptation
Modify preprocessing, model flow, and training/inference pipeline for 2D X-ray compatibility.

⚪ Milestone 3 – Testing, Debugging & Final Delivery
Validate the updated pipeline, resolve issues, and deliver a clean working version of the adapted framework.

Pricing
⚪ My rate for this project is 15$/hour
⚪ Total hours will depend on the current framework complexity and required modifications
⚪ If preferred, I’m also open to discussing a fixed-budget structure after reviewing the codebase and package details

I look forward to learning more about the framework and helping you adapt it efficiently for your X-ray medical imaging application.

Best regards,
Umair

View details
I have changed the terms in this proposal.

View details
SB
Sunshine Bizz
8:11 PM
Thank you for reaching out, here is the ask

Re-engineer the TotalSpineSeg framework to perform automated vertebral segmentation and labeling on 2D X-ray images instead of 3D MRIs.

https://github.com/neuropoly/totalspineseg

GitHub - neuropoly/totalspineseg: Robust Segmentation and Labeling of Vertebrae, Intervertebral Discs, Spinal Cord, and Spinal Canal in MRI Images Using nnU-Net and Iterative Algorithm.
Robust Segmentation and Labeling of Vertebrae, Intervertebral Discs, Spinal Cord, and Spinal Canal in MRI Images Using nnU-Net and Iterative Algorithm. - neuropoly/totalspineseg
GitHub
Umair Ejaz
8:18 PM
ok, let me go through it

Umair Ejaz
8:28 PM
do you have 2D x ray images images?

i would need to take a look those images

I have looked at the repo

TotalSpineSeg is not a generic “spine segmentation framework.” It is a purpose-built pipeline for 3D MRI that does automatic segmentation and labeling of vertebrae, intervertebral discs, spinal cord, and spinal canal. The repo describes a two-step hybrid approach: step 1 predicts semantic spine structures plus landmark discs/C1, then step 2 uses the MRI plus a second channel derived from step 1’s odd-disc output to refine labeling, and a custom iterative algorithm assigns final vertebra/disc IDs.

It also expects NIfTI volumes, supports localizer segmentations when C1 or sacrum is missing, and the inference pipeline explicitly reorients images to LPI and resamples them to 1×1×1 mm before running the models.

The core problem changes when you move from MRI to plain X-ray. MRI is used because it shows soft tissues and the spinal cord well; plain radiography is mainly good for bones and alignment, while soft tissues such as discs, nerves, and the spinal cord are poorly visualized or have only an adjunct role in evaluation.

A few things are reusable:

The overall package structure and training/inference orchestration idea.
The fact that nnU-Net itself supports both 2D and 3D segmentation pipelines.
The concept of landmark-based postprocessing for ordering vertebrae.

The main technical question is the exact target output on X-ray: vertebral body segmentation only, vertebra numbering/labeling, or a broader MRI-style output set. That matters because plain radiographs support vertebral analysis well, but some of the original TotalSpineSeg targets are much more modality-dependent.

If you can share the target X-ray format/views and the available annotations, I can propose the right adaptation path — likely keeping the useful framework structure while redesigning the model and post-processing for a 2D vertebra-focused pipeline.

Umair Ejaz
8:45 PM
I have changed the terms in this proposal.

View details
SB
Sunshine Bizz
10:15 PM
Can you please be concise in your response ?

Brief .... less is better

Umair Ejaz
10:20 PM
Yes — I can re-engineer TotalSpineSeg for a 2D X-ray vertebral segmentation and labeling workflow.

To confirm the best approach, I’d just need to review a few sample X-rays and the available annotations/views. Once I see those, I can define the adaptation path clearly and proceed efficiently.

SB
Sunshine Bizz
10:52 PM
There are millions of x rays available in public data.Sources

Please do your research and let me know

Please don't have chatgpt draft your response

Umair Ejaz
10:53 PM
sure

English is my second language so i just use it for making sentences better

SB
Sunshine Bizz
Mar 30, 2026 | 10:52 PM
Please don't have chatgpt draft your response

Tuesday, Mar 31
Umair Ejaz
6:06 PM
Understood.

I reviewed the framework and the modality shift is feasible, but it is not a direct conversion. TotalSpineSeg is built around 3D MRI volumes and 3D nnU-Net workflows, so for X-rays the practical path is to keep the useful pipeline structure and redesign the model, preprocessing, and labeling logic for 2D radiographs.

There are public spine X-ray datasets available for vertebra segmentation and labeling research, so I can use those to define the adaptation approach and benchmark the pipeline.

My plan would be:

1. identify the best public 2D spine X-ray datasets and annotation type,
2. adapt the framework from 3D volume processing to 2D image processing,
3. rebuild the segmentation/labeling workflow around vertebra-focused X-ray outputs,
4. validate the pipeline on public data before project-specific refinement.

If that matches your expectation, I can outline the implementation steps clearly and start with the framework restructuring.

Wednesday, Apr 01
SB
Sunshine Bizz
4:16 PM
Good morning .... we can do a small POC for $50 that proves to me ... you have the skills and can get the job done

Output of the POC is working Xray output using the model I shared above but for 2d data ...

Full project with trained model using public data is $250

Please let me know your thoughts. Happy to move forward with the above described setup

Umair Ejaz
4:52 PM
let me have discussion with my team, i will get back to you shortly

Umair Ejaz
6:04 PM
I can work with 50$ for PoC and then Full project with trained model using public data will be $250 seperately

if its ok with you, i can start immediately

Umair Ejaz
6:27 PM
also let me know what resources do you have for training

Thursday, Apr 02
Umair Ejaz
4:47 AM
i have been working on PoC

Friday, Apr 03
Umair Ejaz
8:09 PM
im waiting for your response

Saturday, Apr 04
SB
Sunshine Bizz sent an offer

8:51 PM
ML Model for Medical Use-Case X-Ray Focused

The Goal: I need an ML engineer to adapt an existing 3D framework for use with 2D X-ray images

The Skills: You should be experienced in Python and PyTorch, with a understanding of image segmentation

Will share more details and names of existing packages to appropriate candidates, thank you

Est. Budget: $50.00

Milestone 1: Design totalspineseg nnU-Net architecture by using X-Ray public data

Due: Sunday, Apr 5, 2026

Project funds: $25.00

View offer
Umair Ejaz
10:31 PM
thank you so much for the offer

Umair Ejaz
10:58 PM
I have made two variants for milestone 1. Kindly look at both of these and let me know if you want any changes, so I can start with milestone 2

Thanks

2 files
Milestone_1.pdf
102 kB
xray_milestone1_design.pdf
4 kB
SB
Sunshine Bizz
11:20 PM
since you are proving to me you can do the full project, lets move to milestone 2 and get the actual POC done

seems like ChatGPT content?

Umair Ejaz
11:21 PM
I already told you I use it for improving the language, English is not my first language

SB
Sunshine Bizz
Apr 4, 2026 | 11:20 PM
seems like ChatGPT content?

SB
Sunshine Bizz
11:21 PM
k

Umair Ejaz
Apr 4, 2026 | 11:21 PM
I already told you I use it for improving the language, English is not my first language

Umair Ejaz
11:23 PM
im going to start with milestone 2

SB
Sunshine Bizz
11:31 PM
T1 to L5

Content: 609 training and 159 testing images with 18 landmark points for each vertebra
from T1 to L5.

make sure it is covering Cervical ... Thoracic and Lumber

Umair Ejaz
11:35 PM
sure

Sunday, Apr 05
Umair Ejaz
5:07 PM
I completed a first working X-ray adaptation baseline for the TotalSpineSeg workflow on public AP thoracolumbar data (T1-L5).

Current milestone-2 baseline results on the held-out test split:

147 test X-rays evaluated
mean Dice: 0.8560
mean IoU: 0.7525
mean precision: 0.8527
mean recall: 0.8678

What is delivered in this baseline:

2D nnU-Net training pipeline for AP spine X-rays
trained fold-0 checkpoint
inference pipeline for unseen X-rays
postprocessing and visual overlay outputs
evaluation report on held-out data

his public dataset covers thoracic + lumbar (T1-L5) only. It does not include cervical vertebrae, so this baseline is currently thoracolumbar, not full cervical-thoracic-lumbar.

If you want full-spine coverage including cervical levels, the next step is adding a second dataset with C1-C7 annotations and retraining/expanding the pipeline.

metrics_summary.json
metrics_summary.json
621 bytes
You removed this message

here are some previews

01-July-2019-1_jpg.rf.d3d8ccffd35b7e7f478198ceea7b6593.png
01-July-2019-1_jpg.rf.d3d8ccffd35b7e7f478198ceea7b6593.png
01-July-2019-4_jpg.rf.807ec04b7aaf94ab2ce0663575e6d28a.png
01-July-2019-4_jpg.rf.807ec04b7aaf94ab2ce0663575e6d28a.png
2014_ROBERTO_GARIP_22-06-2016_P6220042_JPG_jpg.rf.1003cbaba91e61c79d4df521bdb2f2c0.png
2014_ROBERTO_GARIP_22-06-2016_P6220042_JPG_jpg.rf.1003cbaba91e61c79d4df521bdb2f2c0.png
Umair Ejaz accepted an offer

5:30 PM
View contract
Umair Ejaz requested payment for the milestone

5:31 PM
Milestone :1

Milestone 1: "Design totalspineseg nnU-Net architecture by using X-Ray public data"

Due: Sunday, Apr 05, 2026

Amount: $25.00

View details
Milestone_1 (1).pdf
Milestone_1 (1).pdf
102 kB
SB
Sunshine Bizz approved the milestone

9:02 PM
Milestone 1: "Design totalspineseg nnU-Net architecture by using X-Ray public data"

Due: Sunday, Apr 05, 2026

Amount paid: $25.00

Amount: $25.00

View details
Sunshine Bizz activated the milestone

9:04 PM
Milestone 2: "Development & Testing totalspineseg nnU-Net architecture by using X-Ray public data"

Due: Tuesday, Apr 07, 2026

Amount: $25.00

View details
SB
Sunshine Bizz
9:14 PM
we need the three levels like i mentioned in my chats earlier

please share a working code so i can test it

Umair Ejaz
9:36 PM
yeah sure

SB
Sunshine Bizz
Apr 5, 2026 | 9:15 PM
please share a working code so i can test it

Umair Ejaz
10:26 PM
The public dataset used for this first version does not include cervical annotations, so the current working model is thoracolumbar, not full cervical + thoracic + lumbar yet.

If full cervical + thoracic + lumbar support is required, I can extend this next, but that will require an additional cervical-labeled dataset and another training pass that is very time consuming.

Umair Ejaz requested payment for the milestone

11:17 PM
https://drive.google.com/file/d/1Vv3o-inHTrI_Kpt4ob-6NrHqauyBSOhK/view?usp=drivesdk

Milestone 2: "Development & Testing totalspineseg nnU-Net architecture by using X-Ray public data"

Due: Tuesday, Apr 07, 2026

Amount: $25.00

View details
kindly check

Monday, Apr 06
SB
Sunshine Bizz
2:37 AM
Checking ... thanks

Umair Ejaz
2:39 AM
🙂

Tuesday, Apr 07
SB
Sunshine Bizz
6:27 AM
sorry about the delay ... should have an answer tomorrow

thank you

Umair Ejaz
9:51 AM
ok

Thursday, Apr 09
SB
Sunshine Bizz approved the milestone

3:17 AM
Milestone 2: "Development & Testing totalspineseg nnU-Net architecture by using X-Ray public data"

Due: Tuesday, Apr 07, 2026

Amount paid: $25.00

Amount: $25.00

View details
Umair Ejaz
4:38 AM
Hi

SB
Sunshine Bizz
4:40 AM
hello

Umair Ejaz
4:42 AM
Good Evening! can you please tell me the next steps ?

SB
Sunshine Bizz
4:43 AM
not ready yet for next steps ...

pls give me a few days

I am back-logged with a few projects

Umair Ejaz
4:43 AM
yeah sure
no worries

let me know in the meantime, if you have any work related to me

SB
Sunshine Bizz
8:07 AM
Let's talk tomorrow

Umair Ejaz
8:07 AM
ok

SB
Sunshine Bizz
10:11 PM
i am open to having a direct call and have you explain to me the model / logic / output to accelerate next steps, thank you

Umair Ejaz
10:19 PM
i can send you a video of me explaining the model, i think it will be more better

Friday, Apr 10
SB
Sunshine Bizz
8:06 PM
please do

thank you

Umair Ejaz
8:06 PM
ok

SB
Sunshine Bizz
8:16 PM
what have we accomplished, what is remaining to be proven

what will it take to train the model for all levels of the spine (top to bottom)

what accuracy we can expect

and how close this aligns with totalspineseg

thanks,

Umair Ejaz
8:39 PM
Done so far: Adapted the TotalSpineSeg pipeline for 2D X-rays, trained on the AASCE public dataset (T1–L5). The model segments vertebrae with ~85% Dice on held-out test images.

What's remaining: Cervical levels (C1–C7) are not covered yet , we need a labeled cervical X-ray dataset and another training run to get full spine coverage.

For full C1–L5: I'd need to combine a cervical dataset with the current one and retrain. Depending on GPU resources, training takes 1–3 days.

Expected accuracy: With proper data, ~83–88% Dice is realistic across all levels. Cervical is harder due to anatomy , may be slightly lower initially.

Alignment with TotalSpineSeg: The pipeline structure is similar, nnU-Net backbone, postprocessing for vertebra labeling , but redesigned for 2D and bone-focused (no disc/cord since X-ray doesn't show soft tissue well).

SB
Sunshine Bizz updated the contract

10:23 PM
Milestone 3: Full-Spine Expansion (C1–L5) & Enhanced Validation

ADDED:

Name: Milestone 3: Full-Spine Expansion (C1–L5) & Enhanced Validation

Amount: $50.00

Due: Saturday, Apr 11, 2026

Order: Milestone 3

View Updated Contract
Sunshine Bizz activated the milestone

10:24 PM
Milestone 3: Full-Spine Expansion (C1–L5) & Enhanced Validation

Task: Integrate a cervical X-ray dataset and retrain the 2D nnU-Net model to cover the entire spine (Cervical, Thoracic, and Lumbar).

Deliverable 1: Updated model checkpoint supporting all levels (C1–L5).

Deliverable 2: Performance report on an expanded test set (e.g., >250 images) with Mean Dice scores for each of the three regions.

Deliverable 3: Visual overlay previews showing successful segmentation from the neck down to the lower back.

Milestone 3: "Milestone 3: Full-Spine Expansion (C1–L5) & Enhanced Validation"

Due: Saturday, Apr 11, 2026

Amount: $50.00

View details
we will continue to expand the POC ... based on milestones

Umair Ejaz
10:25 PM
ok sure

SB
Sunshine Bizz
10:42 PM
generate_drr.py
generate_drr.py
11 kB
Umair, Here id an example of how generating synthetic X-rays from 3D CT scans (DRRs). I'm attaching this scripts for generate_drr.py and generate_phantom.py. This may help you to use this logic to create a synthetic Cervical (C1–C7) dataset so we can achieve the 'full spine' coverage we discussed for the next milestone.

This is an example ... not forcing you to one direction or another ... just giving you some help that you may need

generate_phantom.py
generate_phantom.py
6 kB
Umair Ejaz
10:45 PM
ok thanks for sending it, i will look into it

SB
Sunshine Bizz
10:59 PM
for next iteration ... please make this demo ready from command prompt

where we can share series of x-rays (not MRIs) .... and this can pick-up the data and produce valid output

thank you

i have a AWS EC2 env that we can use

or if there is something public, that is an option

Umair Ejaz
11:07 PM
Using your AWS EC2 environment is fine for me, and that would likely be the best option for training and testing at this stage.

For this milestone, I will focus on:

expanding toward cervical coverage,
making the pipeline runnable from command prompt,
preparing outputs that are easy for you to test and review.

SB
Sunshine Bizz
11:08 PM
Accuracy improved

Umair Ejaz
11:09 PM
sure

Saturday, Apr 11
Umair Ejaz
2:54 AM
hi, i have code almost ready for milestone 3, i will need resources for training

Umair Ejaz
3:44 AM
You removed this message

Umair Ejaz
3:53 AM
i hope you understand

SB
Sunshine Bizz
5:16 PM
What resources do you need for training ?

Umair Ejaz
8:46 PM
NVIDIA A10G (24GB VRAM)
instance: g5.xlarge thats the AWS instance

that would be the best choice for us

Umair Ejaz
10:37 PM
Milestone 3 is complete! I've expanded the model to cover the full spine (C1–S1) for both AP and Lateral views.
I did the training on 5,700+ images, which took ~15 hours for 12 epochs, reaching an impressive 0.914 Mean Dice and 0.926 Recall. I used my laptop's GPU RTX3050.

Attached are the full delivery ZIP, metrics, and sample result.

3 files
TotalSpineSeg_XRay_Milestone3_Delivery.zip
330 MB

sample_result_1.png
222 kB
metrics_summary.json
2 kB
Umair Ejaz requested payment for the milestone

10:39 PM
Milestone 3

Milestone 3: "Milestone 3: Full-Spine Expansion (C1–L5) & Enhanced Validation"

Due: Saturday, Apr 11, 2026

Amount: $50.00

View details
sample_result_2.png
sample_result_2.png
244 kB
I have also added README_XRAY.md and requirements.txt for your guide and replication on your end

Umair Ejaz
10:51 PM
per_case_metrics.csv
per_case_metrics.csv
233 kB
Monday, Apr 13
SB
Sunshine Bizz
8:54 PM
Hello Umair, i am getting the following error message

"File could not be retrieved. Large files may take a few moments to process before they become available."

Umair Ejaz
Apr 11, 2026 | 10:37 PM
Milestone 3 is complete! I've expanded the model to cover the full spine (C1–S1) for both AP and Lateral views.
I did the training on 5,700+ images, which took ~15 hours for 12 epochs, reaching an impressive 0.914 Mean Dice and 0.926 Recall. I used my laptop's GPU RTX3050.

Attached are the full delivery ZIP, metrics, and sample result.

Show more
i am unable to download the file ... it is 330 MB

Appreciate your help

Umair Ejaz
8:55 PM
let me upload it on google drive, then i will share the link

SB
Sunshine Bizz
8:59 PM
thank you

image.png
image.png
i am guessing it will use the same structure?

and also, please let me know some details ... how to test it so we know

i will release this milestone once we test this and setup next milestone as needed for additional work

you

Umair Ejaz
9:00 PM
sure, i have added a guide, how you can set it up from your side

let me just share it with you, then i will guide you

SB
Sunshine Bizz
9:03 PM
you can generate that via ChatGPT

Umair Ejaz
9:03 PM
yeah sure

Umair Ejaz
9:13 PM
https://drive.google.com/file/d/16SWR7OVgPa8Tx57jve8dV3FAqiJSMYL5/view?usp=sharing

kindly check it and let me know if it works

in the folder Docs, you will find two guides

SB
Sunshine Bizz
9:19 PM
checking

Umair Ejaz
9:20 PM
ok

Tuesday, Apr 14
SB
Sunshine Bizz
4:29 PM
Good morning

Can we get the full trained nnU-Net model export folder? Thank you

also requiremnt.txt has some local paths so that need to fix ..if that pakages are using

Umair Ejaz
4:32 PM
yeah sure

SB
Sunshine Bizz
8:08 PM
thansk

Wednesday, Apr 15
Umair Ejaz
11:57 AM
https://drive.google.com/file/d/1gaU0NCRajLo3WF4jEm_ZFgHVp5vAeF_d/view?usp=sharing

kindly check it

SB
Sunshine Bizz
6:27 PM
thank you

Thursday, Apr 16
SB
Sunshine Bizz
12:28 AM
Hi, I was able to run the v2 delivery on Linux and inference completes successfully, but the segmentation output looks inaccurate and doesn’t seem anatomically correct on my test case.

I’m attaching a PDF with the input image and the output result for reference.

Could you please confirm whether this model is intended for AP, lateral, or both, and whether any preprocessing is required before inference? If possible, please also share one known-good sample input and expected output from this exact delivery so I can compare.

xay_poc_test.pdf
xay_poc_test.pdf
156 kB
Appreciate your help

Umair Ejaz
1:31 AM
ok

Umair Ejaz
9:12 AM
The current model (Dataset202) was trained for segmentation of the full spine (C1–S1) in both AP and lateral views. It uses a hybrid training dataset composed of both real and synthetic images to support both orientations.

The “scattered blob” predictions seen in your test case are most likely caused by an anatomical scale mismatch between the training data and the input clinical image.

The model was trained on images with a median resolution of approximately 800 × 1000 pixels. Its internal patch size is fixed at 768 × 640.

To address this issue, the input X-ray should be resized so that its long edge is approximately 1024 pixels before running inference. This preprocessing step brings the clinical image closer to the scale seen during training and helps the model interpret the anatomical structures more accurately.

Umair Ejaz
9:22 AM
2 files
sample_result_1.png
sample_result_2.png
For your reference, I have attached two known-good samples from this exact delivery.

SB
Sunshine Bizz
2:37 PM
Thank you

Umair Ejaz
Apr 16, 2026 | 9:25 AM
For your reference, I have attached two known-good samples from this exact delivery.

Can you please share the raw images used to test ?

Umair Ejaz
2:39 PM
lateral sample

0001035_0000.png
0001035_0000.png
AP Sample

0012036_0000.png
0012036_0000.png
these are image used for test that i shared with you

SB
Sunshine Bizz
2:40 PM
Testing

Umair Ejaz
Apr 16, 2026 | 2:40 PM
these are image used for test that i shared with you

Umair Ejaz
2:41 PM
i didnt get it

Umair Ejaz
2:51 PM
do you want all of the images?

SB
Sunshine Bizz
4:55 PM
as many as we can get

Umair Ejaz
5:11 PM
sure

can you tell me what you want to do

i might be able to give some better solution

SB
Sunshine Bizz
5:20 PM
Sunshine Bizz removed this message

Umair Ejaz
5:21 PM
ok

SB
Sunshine Bizz
5:21 PM
checking wtih my team

i will release the milestone soon

and will setup next one to ask for additional help

Umair Ejaz
5:22 PM
sure

Monday, Apr 20
Umair Ejaz
11:29 AM
Hi, any update?

Tuesday, Apr 21
SB
Sunshine Bizz
9:32 PM
Hello

Yes, sharing with you an update later today

Thanks

Umair Ejaz
9:33 PM
ok

Wednesday, Apr 22
SB
Sunshine Bizz
5:01 PM
when you are free, please let me know best time to get on a call and discuss next steps

Umair Ejaz
5:03 PM
we can have meeting tomorrow

let me know when you will be available, i will adjust

SB
Sunshine Bizz
5:08 PM
lets connect around the same time

Umair Ejaz
5:09 PM
sure
