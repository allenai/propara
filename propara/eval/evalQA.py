import sys, collections, pylev
from stemming.porter2 import stem

#--------------------------------------------------------------
# Author: Scott Wen-tau Yih
# Usage: evalQA.py para-ids gold-labels system-predictions
# example usage: python propara/eval/evalQA.py tests/fixtures/eval/para_id.test.txt tests/fixtures/eval/gold_labels.test.tsv tests/fixtures/eval/sample.model.test_predictions.tsv 
#--------------------------------------------------------------

# Data structure for Labels
'''
  PID -> [TurkerLabels]
  TurkerLabels = [TurkerQuestionLabel1, TurkerQuestionLabel2, ... ]  # labels on the same paragraph from the same Turker
  TurkerQuestionLabel -> (SID, Participant, Type, From, To)
'''
TurkerQuestionLabel = collections.namedtuple('TurkerQuestionLabel', 'sid participant event_type from_location to_location')


# Data structure for Predictions
'''
  PID -> Participant -> SID -> PredictionRecord
'''
PredictionRecord = collections.namedtuple('PredictionRecord', 'pid sid participant from_location to_location')

# Fixing tokenization mismatch while alinging participants
manual_participant_map = { 'alternating current':'alternate current', 'fixed nitrogen':'nitrogen',
                           'living things':'live thing', 'red giant star':'star', 'refrigerent liquid':'liquid',
                           'remains of living things':'remains of live thing',
                           "retina's rods and cones":"retina 's rod and cone" } #, 'seedling':'seed'}

#----------------------------------------------------------------------------------------------------------------

def compare_to_gold_labels(participants, system_participants):
    ret = []
    found = False
    for p in participants:
        p = p.lower()
        if p in system_participants:
            ret.append(p)
            continue
        for g in system_participants:
            if (pylev.levenshtein(p,g) < 3):
                #print (p, "===", g)
                ret.append(g)
                found = True
        if not found:
            if p in manual_participant_map:
                ret.append(manual_participant_map[p])
            #else:
            #    print("cannot find", p, system_participants)
    return ret

def preprocess_locations(locations):
    ret = []
    for loc in locations:
        if loc == '-':
            ret.append('null')
        elif loc == '?':
            ret.append('unk')
        else:
            ret.append(loc)
    return ret


def preprocess_question_label(sid, participant, event_type, from_location, to_location, system_participants=None):

    # check if there are multiple participants grouped together
    participants = [x.strip() for x in participant.split(';')]

    # check if there are multiple locations grouped together
    from_locations = preprocess_locations([x.strip() for x in from_location.split(';')])

    # check if there are multiple locations grouped together
    to_locations = preprocess_locations([x.strip() for x in to_location.split(';')])

    #print(participant, participants, system_participants)
    if system_participants != None: # check if the participants are in his list
        participants = compare_to_gold_labels(participants, system_participants)
        #print("legit_participants =", participants)

    #print(from_location, from_locations)
    #print(to_location, to_locations)

    return  [TurkerQuestionLabel(sid, p, event_type, floc, tloc) for p in participants
                                                                 for floc in from_locations
                                                                 for tloc in to_locations]

#----------------------------------------------------------------------------------------------------------------

'''
  Read the 'all-moves' tab in https://docs.google.com/spreadsheets/d/1kWdJFoMnvPDGigV7nxq5tkBFPt_QBHQLbA5seU6wXgY/edit#gid=1762733233
'''
def readLabels(fnLab, selPid=None, gold_labels=None):
    fLab = open(fnLab)
    fLab.readline()    # skip header
    ret = {}
    TurkerLabels = []
    for ln in fLab:
        f = ln.rstrip().split('\t')
        if len(f) == 0 or len(f) == 1:
            if not selPid or pid in selPid:
                if pid not in ret:
                    ret[pid] = []
                ret[pid].append(TurkerLabels)
            TurkerLabels = []
        elif len(f) != 11:
            sys.stderr.write("Error: the number of fields in this line is irregular: " + ln)
            sys.exit(-1)
        else:
            if f[1] == '?': continue
            pid, sid, participant, event_type, from_location, to_location = int(f[0]), int(f[1]), f[3], f[4], f[5], f[6]

            if gold_labels and selPid and pid in selPid:
                #print("pid=", pid)
                TurkerLabels += preprocess_question_label(sid, participant, event_type, from_location, to_location, gold_labels[pid].keys())
            else:
                TurkerLabels += preprocess_question_label(sid, participant, event_type, from_location, to_location)

            #TurkerLabels += (TurkerQuestionLabel(sid, participant, event_type, from_location, to_location))
    return ret

#----------------------------------------------------------------------------------------------------------------

def readPredictions(fnPred):
    ret = {}

    for ln in open(fnPred):
        f = ln.rstrip().split('\t')
        pid, sid, participant, from_location, to_location = int(f[0]), int(f[1]), f[2], f[3], f[4]

        if pid not in ret:
            ret[pid] = {}
        dtPartPred = ret[pid]

        if participant not in dtPartPred:
            dtPartPred[participant] = {}

        dtPartPred[participant][sid] = PredictionRecord(pid, sid, participant, from_location, to_location)

    return ret

#----------------------------------------------------------------------------------------------------------------

def readGold(fn):
    # read the gold label

    dtPar = {}
    for ln in open(fn):
        f = ln.rstrip().split('\t')
        parId, sentId, participant, before_after, labels = int(f[0]), int(f[1]), f[2], f[3], f[4:]

        if (before_after != "before") and (before_after != "after"):
            print("Error:", ln)
            sys.exit(-1)

        if sentId == 1 and before_after == "before":
            statusId = 0
        elif before_after == "before":
            continue  # skip this line
        else:
            statusId = sentId

        if parId not in dtPar:
            dtPar[parId] = {}
        dtPartLab = dtPar[parId]
        if participant not in dtPartLab:
            dtPartLab[participant] = {statusId: labels}
        else:
            dtPartLab[participant][statusId] = labels
    return dtPar

#----------------------------------------------------------------------------------------------------------------

def findAllParticipants(lstTurkerLabels):
    setParticipants = set()
    for turkerLabels in lstTurkerLabels:
        for x in turkerLabels:
            setParticipants.add(x.participant)
    return setParticipants

def findCreationStep(prediction_records):
    steps = sorted(prediction_records, key=lambda x: x.sid)
    #print("steps:", steps)

    # first step
    if steps[0].from_location != 'null':    # not created (exists before the process)
        return -1

    for s in steps:
        if s.to_location != 'null':
            return s.sid
    return -1   # never exists

def findDestroyStep(prediction_records):
    steps = sorted(prediction_records, key=lambda x: x.sid, reverse=True)
    #print("steps:", steps)

    # last step
    if steps[0].to_location != 'null':  # not destroyed (exists after the process)
        return -1

    for s in steps:
        if s.from_location != 'null':
            return s.sid

    return -1   # never exists

def location_match(p_loc, g_loc):
    if p_loc == g_loc:
        return True

    p_string = ' %s ' % ' '.join([stem(x) for x in p_loc.lower().replace('"','').split()])
    g_string = ' %s ' % ' '.join([stem(x) for x in g_loc.lower().replace('"','').split()])

    if p_string in g_string:
        #print ("%s === %s" % (p_loc, g_loc))
        return True

    return False

def findMoveSteps(prediction_records):
    ret = []
    steps = sorted(prediction_records, key=lambda x: x.sid)
    # print(steps)
    for s in steps:
        if s.from_location != 'null' and s.to_location != 'null' and s.from_location != s.to_location:
            ret.append(s.sid)

    return ret

#----------------------------------------------------------------------------------------------------------------

# Q1: Is participant X created during the process?
def Q1(labels, predictions):
    tp = fp = tn = fn = 0.0
    for pid in labels:
        setParticipants = findAllParticipants(labels[pid])
        # find predictions
        be_created = {}
        for participant in setParticipants:
            pred_creation_step = findCreationStep(predictions[pid][participant].values())
            be_created[participant] = (pred_creation_step != -1)
        for turkerLabels in labels[pid]:
            # labeled as created participants
            lab_created_participants = [x.participant for x in turkerLabels if x.event_type == 'create']
            for participant in setParticipants:
                tp += int(be_created[participant] and (participant in lab_created_participants))
                fp += int(be_created[participant] and (participant not in lab_created_participants))
                tn += int(not be_created[participant] and (participant not in lab_created_participants))
                fn += int(not be_created[participant] and (participant in lab_created_participants))
    return tp,fp,tn,fn

# Q2: Participant X is created during the process. At which step is it created?
def Q2(labels, predictions):
    tp = fp = tn = fn = 0.0
    # find all created participants and their creation step
    for pid,lstTurkerLabels in labels.items():
        for turkerLabels in lstTurkerLabels:
            for x in [x for x in turkerLabels if x.event_type == 'create']:
                pred_creation_step = findCreationStep(predictions[pid][x.participant].values())
                tp += int(pred_creation_step != -1 and pred_creation_step == x.sid)
                fp += int(pred_creation_step != -1 and pred_creation_step != x.sid)
                fn += int(pred_creation_step == -1)
    return tp,fp,tn,fn

# Q3: Participant X is created at step Y, and the initial location is known. Where is the participant after it is created?
def Q3(labels, predictions):
    tp = fp = tn = fn = 0.0
    # find all created participants and their creation step
    for pid,lstTurkerLabels in labels.items():
        for turkerLabels in lstTurkerLabels:
            for x in [x for x in turkerLabels if x.event_type == 'create' and x.to_location != 'unk']:
                pred_loc = predictions[pid][x.participant][x.sid].to_location
                tp += int(pred_loc != 'null' and pred_loc != 'unk' and location_match(pred_loc, x.to_location))
                fp += int(pred_loc != 'null' and pred_loc != 'unk' and not location_match(pred_loc, x.to_location))
                fn += int(pred_loc == 'null' or pred_loc == 'unk')
    return tp, fp, tn, fn

#----------------------------------------------------------------------------------------------------------------

# Q4: Is participant X destroyed during the process?
def Q4(labels, predictions):
    tp = fp = tn = fn = 0.0
    for pid in labels:
        setParticipants = findAllParticipants(labels[pid])
        # find predictions
        be_destroyed = {}
        for participant in setParticipants:
            pred_destroy_step = findDestroyStep(predictions[pid][participant].values())
            be_destroyed[participant] = (pred_destroy_step != -1)
        for turkerLabels in labels[pid]:
            # labeled as destroyed participants
            lab_destroyed_participants = [x.participant for x in turkerLabels if x.event_type == 'destroy']
            for participant in setParticipants:
                tp += int(be_destroyed[participant] and (participant in lab_destroyed_participants))
                fp += int(be_destroyed[participant] and (participant not in lab_destroyed_participants))
                tn += int(not be_destroyed[participant] and (participant not in lab_destroyed_participants))
                fn += int(not be_destroyed[participant] and (participant in lab_destroyed_participants))
    return tp,fp,tn,fn

# Q5: Participant X is destroyed during the process. At which step is it destroyed?
def Q5(labels, predictions):
    tp = fp = tn = fn = 0.0
    # find all destroyed participants and their destroy step
    for pid, lstTurkerLabels in labels.items():
        for turkerLabels in lstTurkerLabels:
            for x in [x for x in turkerLabels if x.event_type == 'destroy']:
                    pred_destroy_step = findDestroyStep(predictions[pid][x.participant].values())
                    tp += int(pred_destroy_step != -1 and pred_destroy_step == x.sid)
                    fp += int(pred_destroy_step != -1 and pred_destroy_step != x.sid)
                    fn += int(pred_destroy_step == -1)
    return tp,fp,tn,fn

# Q6: Participant X is destroyed at step Y, and its location before destroyed is known. Where is the participant right before it is destroyed?
def Q6(labels, predictions):
    tp = fp = tn = fn = 0.0
    # find all created participants and their destroy step
    for pid,lstTurkerLabels in labels.items():
        for turkerLabels in lstTurkerLabels:
            for x in [x for x in turkerLabels if x.event_type == 'destroy' and x.from_location != 'unk']:
                pred_loc = predictions[pid][x.participant][x.sid].from_location
                tp += int(pred_loc != 'null' and pred_loc != 'unk' and location_match(pred_loc, x.from_location))
                fp += int(pred_loc != 'null' and pred_loc != 'unk' and not location_match(pred_loc, x.from_location))
                fn += int(pred_loc == 'null' or pred_loc == 'unk')
    return tp, fp, tn, fn

#----------------------------------------------------------------------------------------------------------------

# Q7 Does participant X move during the process?
def Q7(labels, predictions):
    tp = fp = tn = fn = 0.0
    for pid in labels:
        setParticipants = findAllParticipants(labels[pid])
        # find predictions
        be_moved = {}
        for participant in setParticipants:
            pred_move_steps = findMoveSteps(predictions[pid][participant].values())
            be_moved[participant] = (pred_move_steps != [])

        # print(be_moved)

        for turkerLabels in labels[pid]:
            lab_moved_participants = [x.participant for x in turkerLabels if x.event_type == 'move']
            for participant in setParticipants:
                tp += int(be_moved[participant] and (participant in lab_moved_participants))
                fp += int(be_moved[participant] and (participant not in lab_moved_participants))
                tn += int(not be_moved[participant] and (participant not in lab_moved_participants))
                fn += int(not be_moved[participant] and (participant in lab_moved_participants))

    return tp,fp,tn,fn

# Q8 Participant X moves during the process.  At which steps does it move?
def Q8(labels, predictions):
    tp = fp = tn = fn = 0.0
    for pid in labels:
        setParticipants = findAllParticipants(labels[pid])

        # find predictions
        pred_moved_steps = {}
        for participant in setParticipants:
            pred_moved_steps[participant] = findMoveSteps(predictions[pid][participant].values())
        num_steps = len(predictions[pid][participant].keys())

        for turkerLabels in labels[pid]:
            gold_moved_steps = {}
            for x in [x for x in turkerLabels if x.event_type == 'move']:
                if x.participant not in gold_moved_steps:
                    gold_moved_steps[x.participant] = []
                gold_moved_steps[x.participant].append(x.sid)

            for participant in gold_moved_steps:
                res = set_compare(pred_moved_steps[participant], gold_moved_steps[participant], num_steps)
                tp += res[0]
                fp += res[1]
                tn += res[2]
                fn += res[3]
    return tp,fp,tn,fn

def set_compare(pred_steps, gold_steps, num_steps):
    setPred = set(pred_steps)
    setGold = set(gold_steps)
    tp = len(setPred.intersection(setGold))
    fp = len(setPred - setGold)
    fn = len(setGold - setPred)
    tn = num_steps - tp - fp - fn
    return (tp, fp, tn, fn)

# Q9 Participant X moves at step Y, and its location before step Y is known. What is its location before step Y?
def Q9(labels, predictions):
    tp = fp = tn = fn = 0.0
    for pid in labels:
        for turkerLabels in labels[pid]:
            for x in turkerLabels:
                if x.event_type == 'move' and x.from_location != 'unk':
                    pred_loc = predictions[pid][x.participant][x.sid].from_location
                    tp += int(pred_loc != 'null' and pred_loc != 'unk' and location_match(pred_loc, x.from_location))
                    fp += int(pred_loc != 'null' and pred_loc != 'unk' and not location_match(pred_loc, x.from_location))
                    fn += int(pred_loc == 'null' or pred_loc == 'unk')
    return tp,fp,tn,fn

# Q10 Participant X moves at step Y, and its location after step Y is known. What is its location after step Y?
def Q10(labels, predictions):
    tp = fp = tn = fn = 0.0
    for pid in labels:
        for turkerLabels in labels[pid]:
            for x in turkerLabels:
                if x.event_type == 'move' and x.to_location != 'unk':
                    pred_loc = predictions[pid][x.participant][x.sid].to_location
                    tp += int(pred_loc != 'null' and pred_loc != 'unk' and location_match(pred_loc, x.to_location))
                    fp += int(pred_loc != 'null' and pred_loc != 'unk' and not location_match(pred_loc, x.to_location))
                    fn += int(pred_loc == 'null' or pred_loc == 'unk')
    return tp,fp,tn,fn

#----------------------------------------------------------------------------------------------------------------

def main():
    if len(sys.argv) != 4:
        sys.stderr.write("Usage: evalQA.py para-ids gold-labels system-predictions\n")
        sys.exit(-1)
    paraIds = sys.argv[1]
    goldPred = sys.argv[2]
    fnPred = sys.argv[3]
    qid_to_score = {}

    selPid = set([int(x) for x in open(paraIds).readlines()])
    gold_labels = readGold(goldPred)
    labels = readLabels('tests/fixtures/eval/all-moves.full-grid.tsv', selPid, gold_labels)
    predictions = readPredictions(fnPred)

    blHeader = True
    qid = 0
    for Q in [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10]:
        qid += 1
        tp, fp, tn, fn = Q(labels, predictions)
        header,results_str, results = metrics(tp,fp,tn,fn,qid)
        if blHeader:
            print("\t%s" % header)
            blHeader = False
        print("Q%d\t%s" % (qid, results_str))
        qid_to_score[qid] = results[5]

    cat1_score = (qid_to_score[1] + qid_to_score[4] + qid_to_score[7]) / 3
    cat2_score = (qid_to_score[2] + qid_to_score[5] + qid_to_score[8]) / 3
    cat3_score = (qid_to_score[3] + qid_to_score[6] + qid_to_score[9] + qid_to_score[10]) / 4

    macro_avg = (cat1_score + cat2_score + cat3_score) / 3
    num_cat1_q = 750
    num_cat2_q = 601
    num_cat3_q = 823
    micro_avg = ((cat1_score * num_cat1_q) + (cat2_score * num_cat2_q) + (cat3_score * num_cat3_q)) / \
                (num_cat1_q + num_cat2_q + num_cat3_q)

    print("\n\nCategory\tAccuracy Score")
    print("=========\t=====")
    print(f"Cat-1\t\t{round(cat1_score,2)}")
    print(f"Cat-2\t\t{round(cat2_score,2)}")
    print(f"Cat-3\t\t{round(cat3_score,2)}")
    print(f"macro-avg\t{round(macro_avg,2)}")
    print(f"micro-avg\t{round(micro_avg,2)}")

def metrics(tp, fp, tn, fn, qid):
    if (tp+fp > 0):
        prec = tp/(tp+fp)
    else:	 	
        prec = 0.0
    if (tp+fn > 0):
        rec = tp/(tp+fn)
    else:		
        rec = 0.0
    if (prec + rec) != 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0.0
    accuracy = (tp+tn) / (tp + fp + tn + fn)
    if qid == 8:
        accuracy = f1   # this is because Q8 can have multiple valid answers and F1 makes more sense here
    total = tp + fp + tn + fn

    header = '\t'.join(["Total", "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1"])
    results = [total, tp, fp, tn, fn, accuracy*100, prec*100, rec*100, f1*100]
    results_str = "%d\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f" % (total, tp, fp, tn, fn, accuracy*100, prec*100, rec*100, f1*100)
    return (header, results_str, results)

#----------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
