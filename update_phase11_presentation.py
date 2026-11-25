"""Append Phase-11 slides into the original AA 272 deck using editable PPTX shapes."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT=Path(__file__).resolve().parent
src=ROOT/'FINAL_DELIVERABLES'/'FINAL_RESEARCH_STORY.pptx'
out=ROOT/'FINAL_DELIVERABLES'/'FINAL_RESEARCH_STORY_WITH_KALMAN.pptx'
figs=ROOT/'results'/'phase11'
prs=Presentation(src)
navy=RGBColor(20,39,53);teal=RGBColor(34,113,116);muted=RGBColor(77,100,113)
gray=RGBColor(234,241,242)


def txt(slide,x,y,w,h,text,size=18,color=navy,bold=False,align=None):
    box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=box.text_frame;tf.clear();tf.word_wrap=True
    p=tf.paragraphs[0];p.text=text;p.space_before=Pt(0);p.space_after=Pt(0)
    if align:p.alignment=align
    for run in p.runs:
        run.font.name='Aptos';run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=color
    return box

def base(title,sub,number):
    slide=prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid();slide.background.fill.fore_color.rgb=RGBColor(255,255,255)
    line=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,Inches(.12))
    line.fill.solid();line.fill.fore_color.rgb=teal;line.line.fill.background()
    txt(slide,.65,.38,12.1,.61,title,28,navy,True)
    txt(slide,.67,1.06,12,.35,sub,15,muted)
    line=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(.55),Inches(7.00),Inches(12.2),Inches(.012))
    line.fill.solid();line.fill.fore_color.rgb=RGBColor(185,207,210);line.line.fill.background()
    txt(slide,.67,7.07,11,.25,'STANFORD AA 272  •  MEASURED GNSS + NFL SIMULATION',9,muted)
    txt(slide,12.45,7.07,.38,.25,str(number),9,muted,align=PP_ALIGN.RIGHT)
    return slide

def callout(slide,x,y,w,h,head,body):
    card=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    card.fill.solid();card.fill.fore_color.rgb=gray;card.line.color.rgb=RGBColor(220,232,232)
    txt(slide,x+.22,y+.18,w-.38,.44,head,17,teal,True)
    txt(slide,x+.22,y+.70,w-.40,h-.75,body,14,navy)

def picture(slide,file,x,y,w,h):
    slide.shapes.add_picture(str(file),Inches(x),Inches(y),width=Inches(w),height=Inches(h))

def notes(slide,text):
    if slide.has_notes_slide:
        slide.notes_slide.notes_text_frame.text=text

s=base('Kalman filtering: what does it add?','Phase 11 uses the same Doppler observations as the original stadium experiment',14)
callout(s,.70,1.72,3.77,3.52,'Single-epoch WLS','Solves velocity from the satellites available at one time. If a valid solution is unavailable, that epoch is rejected.')
callout(s,4.78,1.72,3.77,3.52,'Causal Kalman filter','Predicts velocity from the prior state, then uses Doppler measurements to update it. No NFL reference velocity enters the filter.')
callout(s,8.87,1.72,3.77,3.52,'Prediction-only tracking','During a brief outage the filter can report a predicted speed. The formal uncertainty grows; a prediction is not another GNSS measurement.')
txt(s,.78,5.54,11.7,.7,'Tested: constant-velocity and constant-acceleration models • 4/6-satellite updates • process-noise sensitivity',16,teal,True)
notes(s,'Kalman filtering was implemented as a direct Doppler filter with a causal state prediction and Joseph covariance update. Every filter initializes from a valid six-satellite WLS estimate. Fixed process-noise values were selected before evaluating the held-out games. Satellite updates and prediction-only outputs are reported separately.')

s=base('Fixed roofs: Kalman does not fix satellite visibility','Six paired seeds, five held-out NFL games, identical Doppler error draws',15)
picture(s,figs/'11_kalman_event_recall.png',.75,1.60,11.8,4.45)
txt(s,.9,6.24,11.8,.53,'Open sky recall: WLS 47.97%  •  CV-KF 47.39%  •  CA-KF 48.55%',17,teal,True)
notes(s,'In the original static sky masks, satellite geometry remains fixed along each short player track. If a filter cannot initialize from six satellites, it has no previous state to propagate. In the fixed deep-canopy condition WLS6 recall was 13.41%, CV-KF6 or CV-KF4 was also 13.41%, and CA-KF4 was 13.53%. Four-satellite WLS obtains higher recall under fixed obstruction, but uses a different solution requirement and has no residual redundancy at exactly four satellites.')

s=base('Prediction can fill an outage — but may miss a sprint','Temporary 1- and 2-second signal losses are separate hypothetical sensitivity tests',16)
picture(s,figs/'11_gnss_update_vs_prediction.png',.65,1.62,8.0,4.55)
callout(s,8.88,1.90,3.64,2.03,'Two-second outage','WLS: 42.71% valid epochs\nCV-KF: 100% reported estimates, including predictions')
callout(s,8.88,4.15,3.64,1.78,'Detection did not improve','Event recall: WLS 19.92%\nCV-KF 19.69%')
notes(s,'Every fourth sample cycle experiences one or two seconds of deliberately modeled complete GPS unavailability, after an open-sky initialization. During two-second gaps CV-KF predictions increase reported coverage but also increase conditional speed RMSE from 1.03 m/s for updated WLS solutions to 2.79 m/s for the Kalman mixture of updates and predictions. This does not recover movements that were never observed by the 1-Hz receiver.')

s=base('The four-satellite control changes the interpretation','Under canopy entry, additional usable measurements matter more than temporal filtering',17)
picture(s,figs/'11_intermit_event_recall.png',.67,1.52,8.35,4.72)
callout(s,9.22,1.86,3.44,2.07,'Enter deep canopy','WLS6: 13.76% recall\nWLS4: 38.62% recall\nCV-KF4: 38.04% recall')
callout(s,9.22,4.21,3.44,1.88,'Main inference','The jump from six to four satellites is the major change, not the Kalman recursion.')
notes(s,'A fair control is WLS that can estimate velocity from four GPS satellites. A four-satellite WLS solution cannot independently check residual consistency at exactly four measurements, so it also has limitations and false alerts increase under heavy obstruction. In the canopy-entry experiment the Kalman filter starts under an open-sky six-satellite epoch and later updates with four satellites. Its event recall is close to WLS4, demonstrating little independent temporal-estimation advantage. A constant-acceleration KF produces small positive differences in the five-game cohort but no general superiority claim is made.')

# Move four new slides directly before original conclusions and references.
ids=prs.slides._sldIdLst
moving=[ids[-4],ids[-3],ids[-2],ids[-1]]
for node in moving:ids.remove(node)
for offset,node in enumerate(moving):ids.insert(13+offset,node)
# Update slide-number footer fields in the two original slides now at positions 17 and 18.
for index,number in [(17,'18'),(18,'19')]:
    slide=prs.slides[index]
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text.strip() in ('14','15'):
            shape.text_frame.paragraphs[0].text=number
# Update original conclusions to mention the new finding, without removing its cautions.
conclusions=prs.slides[17]
for shape in conclusions.shapes:
    if shape.has_text_frame and shape.text.strip()=='WLS, FDE and robust WLS behave similarly in the paired roof experiments.':
        shape.text_frame.paragraphs[0].text='WLS, FDE, robust WLS, and causal Kalman filtering show that missing satellite updates remain the main constraint.'
    if shape.has_text_frame and shape.text.strip()=='Visibility and sampling control coverage':
        shape.text_frame.paragraphs[0].text='Visibility and sampling control coverage'

prs.save(out)
print('Saved',out,'slides',len(prs.slides))
