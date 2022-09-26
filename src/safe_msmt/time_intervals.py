from interval import interval
import numpy as np

# get usable segments from available interval lists
def seg_available(available, clip_length, neg = False):
    seg = []
    for a1, a2 in available:
        length = a2 - a1
        if length <= clip_length:
            if neg == False:
                seg.append([a1, a2])
        else:
            sg = seg_long_calls(a1, a2, clip_length, neg)
            for s in sg:
                seg.append(s)
    return seg


#split long calls into segments of equal length
#the last segment will have overlap with previous segment if too short
#ensures the last segment will contain at least 1 second of call
# if negative segments only segments longer than 1 seconds will be added to the list with overlap
def seg_long_calls(start, end, clip_length, neg = False):
    length = end - start
    num_clips = int(np.ceil(length / clip_length))
    segments = []
    for i in range(num_clips):
        s = start + i*clip_length
        if i != num_clips - 1:
            e = s + clip_length
            segments.append([s, e])
        else:
            e = end
            if neg == False and e - s < 1:
                s = e - 1
                segments.append([s, e])
            if neg == True and e - (start + i*clip_length) >= 1:
                s = e - clip_length
                segments.append([s, e])
    return segments


def get_duration(interv):
    return interv[0][1] - interv[0][0]


# require sorted overlap
def merge_overlaps(overlap):
    if len(overlap) <= 1:
        return overlap
    new = []
    i = 1
    while i < len(overlap):
        o = overlap[i] & overlap[i - 1]
        if o != interval():
            start = overlap[i - 1][0][0]
            if overlap[i][0][1] >= overlap[i - 1][0][1]:
                end = overlap[i][0][1]
            else:
                end = overlap[i - 1][0][1]
            new.append(interval([start, end]))
            if i + 2 < len(overlap):
                i += 2
            else:
                i += 1
        else:
            new.append(overlap[i - 1])
            if i == len(overlap) - 1:
                new.append(overlap[i])
            i += 1

    return new


# get rid of overlap in overlaps (duplicates)
def clean_overlap(overlap):
    ol = overlap
    for o1 in overlap:
        for o2 in overlap:
            if o1 != o2:
                if o1 in o2 and o1 in ol:
                    ol.remove(o1)
                elif o2 in o1 and o2 in ol:
                    ol.remove(o2)
    return ol


# a merge sort function to sort the overlap list
def sort_overlap(overlap):
    if len(overlap) <= 1:
        return overlap

    left = []
    right = []
    for i in range(len(overlap)):
        if i < np.ceil(len(overlap) / 2):
            left.append(overlap[i])
        else:
            right.append(overlap[i])

    left = sort_overlap(left)
    right = sort_overlap(right)

    return merge(left, right)


def rest(li):
    if len(li) > 1:
        return li[1:]
    else:
        return []


def correct_order(intv1, intv2):
    s1 = intv1[0][0]
    s2 = intv2[0][0]
    if s1 > s2:
        return False
    return True


def merge(left, right):
    result = []
    while len(left) > 0 and len(right) > 0:
        if correct_order(left[0], right[0]):
            result.append(left[0])
            left = rest(left)
        else:
            result.append(right[0])
            right = rest(right)

    while len(left) > 0:
        result.append(left[0])
        left = rest(left)
    while len(right) > 0:
        result.append(right[0])
        right = rest(right)

    return result


def notinrange(overlap, thiscall):
    for o in overlap:
        if thiscall in o or o in thiscall:
            return False
    return True


def find_overlaps(file, start, end, y_labels, filenegative):
    durationdic = {}
    labelsdic = {}
    totaldic = {}
    # calculate the overlap for each file
    for x in np.unique(file):
        val = []
        y = []
        length = 0
        for fi, s, e, cls in zip(file, start, end, y_labels):
            if fi == x and cls != 'nothing':
                i = interval([s, e])
                length += get_duration(i)
                val.append(i)
                y.append(cls)
        if x not in filenegative:
            durationdic[x] = val
            labelsdic[x] = y
            totaldic[x] = length

    # get the overlaps
    overlap = {}
    overlapped_duration = {}
    overlapcls = {}
    for f in np.unique(file):
        if f not in filenegative:
            x = durationdic[f]
            o = []
            sth = interval()
            clss = labelsdic[f]
            oc = []
            for i in range(len(x) - 1):
                cls = clss[i + 1]
                sth = sth | x[i]
                result = x[i + 1] & sth
                if result != interval():
                    o.append(result)
                    oc.append(cls)
            overlap[f] = o
            overlapcls[f] = oc
            overlapped = 0

            ol = merge_overlaps(sort_overlap(clean_overlap(o)))
            for iv in ol:
                overlapped += get_duration(iv)
            if overlapped > 45:
                overlapped = 45
            overlapped_duration[f] = overlapped

    return overlap, overlapcls, overlapped_duration, totaldic


def find_overlaps_v2(file, start, end, y_labels, filenegative, unique_classes):
    durationdic = {}
    labelsdic = {}
    totaldic = {}
    # calculate the overlap for each file
    for x in np.unique(file):
        val = []
        y = []
        length = 0
        for fi, s, e, cls in zip(file, start, end, y_labels):
            if fi == x and cls != 'nothing' and cls in unique_classes:
                i = interval([s, e])
                length += get_duration(i)
                val.append(i)
                y.append(cls)
        if x not in filenegative:
            durationdic[x] = val
            labelsdic[x] = y
            totaldic[x] = length

    # get the overlaps
    overlap = {}
    overlapped_duration = {}
    overlapcls = {}
    for f, cls in zip(np.unique(file), y_labels):
        if cls not in unique_classes:
            continue
        if f not in filenegative:
            x = durationdic[f]
            o = []
            sth = interval()
            clss = labelsdic[f]
            oc = []
            for i in range(len(x) - 1):
                cls = clss[i + 1]
                sth = sth | x[i]
                result = x[i + 1] & sth
                if result != interval():
                    o.append(result)
                    oc.append(cls)
            overlap[f] = o
            overlapcls[f] = oc
            overlapped = 0

            ol = merge_overlaps(sort_overlap(clean_overlap(o)))
            for iv in ol:
                overlapped += get_duration(iv)
            if overlapped > 45:
                overlapped = 45
            overlapped_duration[f] = overlapped

    return overlap, overlapcls, overlapped_duration, totaldic


def find_nonoverlapping_interval(overlap,
                                 thiscall,
                                 minimum):
    if len(overlap) == 0:
        return thiscall

    if notinrange(overlap, thiscall):
        return thiscall

    start = thiscall[0][0]
    end = thiscall[0][1]
    available = []

    # sort and get rid of duplicates
    ol = merge_overlaps(sort_overlap(clean_overlap(overlap)))

    for i in range(len(ol)):
        s = ol[i][0][0]
        e = ol[i][0][1]
        if s - start >= minimum and len(available) == 0:
            if i == 0:
                available.append([start, s])
            elif i > 0:
                x = ol[i - 1] & interval([start, s])
                if x == interval():
                    available.append([start, s])
        if i > 0 and s - ol[i - 1][0][1] >= minimum and ol[i - 1][0][1] >= start and s <= end:
            available.append([ol[i - 1][0][1], s])
        if i == len(ol) - 1 and end - e >= minimum:
            available.append([e, end])
    return available

