# GRITSENSE: Viva Questions and Answers

All numbers are from the test clip: 2022 FIVB Men's World Championship, Brazil v Japan,
319 frames at 30 fps (10.6 s).

---

## A. Overview

**Q: What does your project do?**
It turns a broadcast volleyball video into live tactical advice. It tracks every player,
their pose and the ball; converts everything to real court metres through the court
homography; works out where and how the attacking team should hit; and shows this on a
dashboard with key matchups, team shape and AI-written commentary.

**Q: What are the main stages?**
1. Perception: player tracking, pose, ball detection.
2. Homography and camera: image pixels → court metres; the camera itself is recovered.
3. Ball cleaning and touch detection: the real ball path, and who touches it when.
4. Graph learning: interaction graphs → GAT → HRN (trained without manual labels).
5. Physics EPV: where to hit and with which shot.
6. Hybrid: physics combined with the trained GAT/HRN.
7. Commentary (LLM) and the dashboard video.

**Q: What is the system based on?**
Computer vision (YOLO detection, tracking, pose), projective geometry (homography and
camera recovery), ball-flight physics (gravity, air drag, net clearance), "pitch control"
modelling from football analytics, graph neural networks (GAT, HRN), and large language
models for commentary.

---

## B. Detection, homography and camera

**Q: How are players detected?**
YOLO (`player_detection.pt`) with the BoT-SORT tracker; a YOLO pose model gives 17 body
keypoints, matched to each tracked player. People more than 2 m from the court lines
(bench, referees) are dropped, keeping at most 6 per team.

**Q: What is the homography?**
A 3 × 3 matrix mapping the court floor in the image to real court metres. The four court
corners in `annotations.json` are matched to (0, 0), (18, 0), (18, 9) and (0, 9) metres.
Positions use the players' feet, because only points on the floor map correctly.

**Q: What else did you get from the homography?**
The camera itself. With square pixels and the principal point at the image centre, a
floor homography fixes the camera up to its focal length. One focal length (2,500 px) is
fitted to all frames; then each frame gives the camera's position and angle
(`analytics/camera.py`). The camera measures heights in metres from a single image.

**Q: How do you know the camera is right?**
- Its position stays within about 1 m across all 319 frames while the broadcast camera
  pans: about 33 m behind the far sideline, 10 m up, level with the net.
- Drawing the net (2.43 m) and antennas back into the video lands them on the real ones.
- Standing players measure a median 1.92 m tall (2,636 measurements), as national-team
  men are.

**Q: What does the camera measure?**
Jump heights (81 jumps; 90% of jumps near the net reach 0.70 m or less, maximum 1.00 m),
reach heights (highest 3.19 m), heights of ball touches, and each jumping player's true
court position (their take-off point, because airborne feet look farther away).

**Q: How do you detect a jump?**
Within 0.33 s, the feet and head rise together while the box keeps its size. Walking away
from the camera also lifts the feet in the image but shrinks the box. I confirmed a
detected jump by eye on the video (frame 272).

**Q: How is player movement measured?**
Velocity is the slope of a straight-line fit to each player's court positions over 0.2 s
(6 frames), capped at the measured top speed; position is the mean of the last 3 frames.

---

## C. Ball tracking and touches

**Q: What was wrong with the original ball track?**
It kept one detection per frame and often locked onto the wrong thing: the first
referee's bald head, the mouse cursor (the video is a screen recording) and the net's
antenna. I checked these by eye on the video.

**Q: How did you fix it?**
1. Re-detect with `ball.pt` at 640, 1280 and 1920 px, keeping about 4 candidates per frame
   (the ball is tiny at the default 640 px).
2. Remove static objects: spots detected again and again without moving, with no player
   there to hold a ball.
3. Choose the physical path by dynamic programming, with a speed limit from the camera
   (the fastest spike from the closest court point moves at most ~99 px per frame).
4. Fill gaps of up to 8 frames with a quadratic fit (a flight is a parabola over short
   spans).

**Q: How much better is it?**
Real detections in 259 frames plus 58 filled, so the ball is known in 317 of 319 frames,
following clean flight arcs. Before, the ball appeared in 229 frames, many of them false.
The graph pipeline gets 317 graphs instead of 229.

**Q: How do you detect a touch?**
A sharp bend in the ball's path (velocity over the next 3 frames differs from the previous
3 by more than 20 px/frame) with a player within reach, checked in metres through the
camera: the ball between the floor and 3.67 m (published highest reach plus a 0.3 m
margin) above them, and within 1.5 m sideways.

**Q: How do you know what kind of touch it is?**
From the camera's height measurement and where the ball goes next:
- **Set:** the ball is sent up from above head height (≥ 2.2 m).
- **Pass or dig:** sent up from lower.
- **Attack:** high, sent across or down, **and the ball reaches the other side of the net
  within 1.5 s** (an attack always goes over).

**Q: How accurate is it?**
10 touches detected; checking each on the video, 7 are real and 3 are false (the ball,
high over the far side, lines up with a nearer player in the image). Every real touch is
classified correctly: sets (frames 71, 176, 255), passes and digs (130, 213, 273, 308).
The over-the-net check turns two of the false ones into harmless "passes", so only one
(frame 296) briefly shows "where to hit" when it shouldn't.

**Q: What are touches used for?**
- **Possession:** a touch decides which team has the ball.
- **The hitter:** the player on the ball.
- **The rally phase:** pass → set → attack.
- **When "where to hit" is shown** (section E).

**Q: Can you measure ball speed from the video?**
I tried, starting each flight at a touch where the 3-D position is known. It failed: from
a high broadcast camera, a ball moving away and a ball rising look almost identical, and
30 fps with small camera-pan errors can't separate them. Published speeds are used.

**Q: What did the touches measure?**
Sets at 2.38 m and 2.59 m (matching the 2.48 m standing reach) and the highest play at the
net at 3.19 m, matching the published 3.18 m block reach. Set to attack took 0.6 s.

---

## D. Physics EPV (where to hit)

**Q: What is EPV?**
Expected Point Value: the probability that hitting the ball to a spot wins the point
outright.

**Q: How is it calculated?**
For every 0.25 m spot on the defending half (1,296 spots) and every shot the attacker can
legally play:

```
EPV = P(ball lands in) × (1 − P(blocked)) × P(no defender reaches it)
```

**Q: How do you get the flight time?**
By simulating the flight: for each launch speed in the shot's published range and every
angle, the ball flies from the contact height under gravity and air drag (published ball
mass 0.269 kg, radius 0.105 m, drag coefficient 0.17 at match speeds). A spot counts only
if a trajectory lands there and clears the net (2.43 m + ball radius). The fastest such
trajectory is used.

**Q: How do you get "no defender reaches it"?**
A race between ball and defender ("pitch control"):
- Defender time = effective reaction (0.33 s) + time to land if in the air (√(2h/g)) +
  run at the measured top speed (4.75 m/s) to within dive reach (0.96 m).
- P(reach) = sigmoid((ball time − defender time) / 0.109 s); P(no one) = product over
  defenders.

**Q: Where does the 0.33 s reaction come from?**
Published in-game reaction of elite players (0.571 s, a light-signal test) minus the
published anticipation lead: defenders still read the attack from video cut 240 ms
before contact. It lies inside the published range of 0.26–1.06 s.

**Q: How is the block modelled?**
Attacks are judged **at the moment of the hit**. The 3 front-row defenders nearest the net
step to it, move along it toward the hitter during the set (measured set time 0.6 s,
measured top speed, after reacting), and jump to the published block reach (3.18 m). The
ball is blocked if it crosses the net below a blocker's hands and within their hands'
width.

**Q: Why judge blockers at the hit?**
Judged as they stood (usually not jumping), the block chance was only 3%, so the advice
was always the deep middle (Zone 6 in 80% of frames). Real blockers jump to meet the
attack. With blockers at the hit, the advice plays around the block: line, cross-court,
tips over it, and all six zones.

**Q: How do you get "lands in"?**
The landing point has an aim error (normal distribution) whose spread is set so serves
miss at published rates (20.9% jump serves, 10.8% floats): 1.66 m and 0.85 m.

**Q: Where does every constant come from?**
Listed in `analytics/constants.py`:

| Source | Constants |
|---|---|
| **Measured** from the video | top speed 4.75 m/s, stature 1.92 m, block jump 0.70 m, set time 0.6 s |
| **Published** | net 2.43 m (FIVB); ball mass, size and drag; spike 106.3 ± 5.8 km/h; jump serve 23.0 ± 3.9 m/s; jump float 12.1 ± 3.4 m/s; contact height 3.37 m and block reach 3.18 m (FIVB 2022 team bulletin, Brazil); reaction 0.571 s; anticipation 240 ms; serve error rates |
| **Derived** | reaction 0.33 s, dive reach 0.96 m (half the arm span), standing block reach 2.48 m, race spread 0.109 s, aim spreads |
| **Estimated** (no source found) | tip speed cap, blocker hand width ±0.5 m, hand-height spread 0.1 m, spike aim error equal to the jump serve's |

**Q: How did you validate it?**
Against published attack success, without fitting to it: in the 2010 Men's World
Championship (2,925 attacks), terminal attacks succeeded 72.5% for outside hitters and
55.1% for opposites. On the clip, a perfectly placed attack averages 88% and one aimed
anywhere legal averages 39%. Real elite attacks fall between those, as they should.

**Q: Why physics and not machine learning for EPV?**
A learned EPV needs thousands of rallies labelled with who won the point; there is one
10-second clip. Physics needs no labels, explains itself ("ball 0.22 s vs defender
0.63 s"), and can later label data for a learned model.

---

## E. Moves and when advice is shown

**Q: How many moves can it recommend?**
Five shot types, six recommendations:

| Recommendation | Launch speed (published) | Played from |
|---|---|---|
| Spike · down the line | 27.9–31.1 m/s | within 3 m of the net |
| Spike · cross-court | 27.9–31.1 m/s | within 3 m of the net |
| Tip / roll shot | 5–12 m/s, fastest arc over the block | within 3 m of the net |
| Back-row attack | 27.9–31.1 m/s | behind the 3 m line |
| Jump serve | 19.1–27.0 m/s | at the baseline |
| Float serve | 8.6–15.5 m/s | at the baseline |

Where each can land comes from the flight physics, not from fixed rules.

**Q: When is "where to hit" shown?**
Only when a hit over the net is coming: a **serve**, a **set** to a hitter, or the
**attack** itself. During passes and digs, and while the ball crosses the net, the
dashboard shows the build-up ("PASS · SET TO #52") with no target or map, and the
commentary waits.

**Q: How is the hitter chosen on a pass or set?**
The teammate whose position gives the best possible attack. For example, Japan: #1 digs →
"set to #52", #76 sets → "set to #3".

**Q: Does the advice change during play?**
Yes. In the 112 frames with an attack chance: back-row attack 33, cross-court 38, down the
line 22, tip 19, spread over all six zones.

**Q: Why doesn't the advice flicker?**
A new move must stay best for 10 frames (immediately if the hitter changes), and the target
is smoothed and snapped to the best reachable spot within 1 m.

**Q: What do the dashboard numbers mean?**
- **Point chance:** EPV at the target.
- **Block chance:** chance the block stops the shot.
- **Out chance:** chance the shot lands out.
- **"Ball 0.22 s vs defender 0.63 s":** the ball's flight time to the target against the
  nearest defender's reach time.

---

## F. GAT, HRN and the hybrid

**Q: What is your graph?**
Nodes are players and the ball. Each node has 49 features: type, position, velocity, team,
17 pose keypoints × 2, pose availability, confidence and court features. Edges are
player–player, player–ball and ball–player, each with relative x, relative y, distance and
same-team.

**Q: What are GAT and HRN?**
GAT (Graph Attention Network) learns how much each player's node attends to each
neighbour. HRN (Hierarchical Relational Network, after Ibrahim & Mori, ECCV 2018) builds
player → team → whole-scene representations.

**Q: How were they trained without labels?**
- **GAT, self-supervised:** predict each player's movement 0.5 s ahead; the tracking gives
  the answer.
- **HRN, distillation:** learn the physics model's best zone, shot, threat and team shape.
- Split into 1 s blocks: 3 training, 1 validation (to choose the epoch), 1 test.

**Q: What were the results?**
On the held-out test set:

| Model | Result | Simple baseline |
|---|---|---|
| GAT movement | 0.87 m error | 0.76 m (assume no one moves) |
| HRN zone | 46% | 49% (always the most common zone) |
| HRN shot | 92% | 92% |
| HRN team shape | 0.066 error | — |

They don't beat their simplest baselines yet. One 10-second clip is far too little data;
the training curves show training loss falling while validation stays flat.

**Q: How are they combined with physics?**
- **Key matchups:** 70% physics (how much threat each defender stops, by removing them and
  re-running the model) + 30% GAT attention.
- **Team shape:** 70% physics coverage and balance + 30% HRN.
- **Where to hit:** physics, re-ranked by the HRN's zone and shot beliefs, **weighted by
  the HRN's measured skill on the test set**. Its skill is currently zero, so it doesn't
  move the target; otherwise its "always Zone 6" habit made the advice constant. As more
  data improves it, its influence rises automatically.

**Q: Is your HRN based on published work?**
Yes: Ibrahim & Mori, "Hierarchical Relational Networks for Group Activity Recognition and
Retrieval", ECCV 2018. Related graph work: Actor Relation Graphs (CVPR 2019) and MP-GCN
(ECCV 2024), which also uses players' skeletons and the ball. COMPOSER (ECCV 2022) is the
only one I found with published volleyball weights.

---

## G. Commentary

**Q: How is commentary generated?**
Rules decide when to speak (possession change, new recommended move during an attack
chance, periodic insight); an LLM writes the sentence from computed facts.

**Q: Which LLMs?**
Claude (`claude-opus-5-5`) through the Anthropic API, or Qwen 2.5 1.5B Instruct running
locally, with fixed templates as the fallback.

**Q: How do you stop it from making things up?**
It only receives computed facts, never the video, and each line must mention the real
zone and player number or the template line is used.

---

## H. Limitations and future work

**Q: What are the limitations?**
1. One 10-second clip: the GAT and HRN don't beat their baselines, and the set time comes
   from one set-to-attack pair.
2. Four constants have no source (listed as estimates).
3. Contact and block heights come from the FIVB bulletin's Brazil figures, seen in a
   search result; the bulletin itself couldn't be opened.
4. 3 of 10 detected touches are false (one of them still shows "where to hit" briefly),
   and touch classification looks about 0.1 s ahead (fine for recorded video, a small
   delay if live).
5. Ball speed, spin and 3-D path can't be measured from this video.
6. Player numbers are tracker IDs, not shirt numbers; court corners are annotated by hand.
7. The Claude commentary path is untested here (no API key); the local LLM is slow.

**Q: What would you do next?**
More match videos (no code changes needed), training on the Volleyball Dataset, rallies
labelled with where the ball landed and who won, automatic court detection, and jersey
number recognition.

**Q: What is the biggest weakness?**
Data. The pipeline is complete and its physics is sourced and validated, but one short
clip can't make the learned models useful or prove accuracy statistically.
