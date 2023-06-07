import cv2
import numpy as np
import math
from mmpose.core.post_processing import transform_preds

def topK_sort(dis, anchor, topK=4):
    # import pdb
    # pdb.set_trace()
    
    dis_new=[]
    anchor_new=[]
    length=len(dis)-1

    sorted_nums = sorted(enumerate(dis), key=lambda x: x[1])
    for i in range(topK):
        dis_new.append( sorted_nums[length-i][1] )
        anchor_new.append( anchor[sorted_nums[length-i][0] ] )
    
    return dis_new, anchor_new

def topK(heatmaps, anchor, topK=[0,5] ):
    N, K, H, W = heatmaps.shape

    anchor_expand = np.zeros( (N, K, H*W, 2) )
    anchor_expand[:,:,] = anchor

    heatmaps_trans = np.reshape(heatmaps, (N,K,-1))
    # heatmaps_trans_map = np.argsort(heatmaps_trans) # zhengxu
    heatmaps_trans_map = np.argsort(-heatmaps_trans) # fanxu
    
    indices = np.indices((N,K,H*W))
    indices[2,:,:,:] = heatmaps_trans_map
    heatmaps_trans_sort = heatmaps_trans[indices[0], indices[1], indices[2]]
    heatmaps_trans_sort=heatmaps_trans_sort[:,:,topK[0]:topK[1]]

    anchor_sort = anchor_expand[indices[0], indices[1], indices[2]]
    anchor_sort = anchor_sort[:,:,topK[0]:topK[1]]

    return heatmaps_trans_sort, anchor_sort

##------------------------------------------
def block_search(hm, center, H, W, step_h=2, step_w=3, gl=False):
    # import pdb
    # pdb.set_trace()
    
    centerX=int(center[0]+0.5)
    centerY=int(center[1]+0.5)
    Ymin=max(centerY-step_h, 0)
    Ymax=min(centerY+step_h+1, H)
    Xmin=max(centerX-step_w, 0)
    Xmax=min(centerX+step_w+1, W)
    
    if gl:
        Ymin=0
        Ymax=H
        Xmin=0
        Xmax=W

    max_val=-99999999.0
    bbox=[Ymin, Ymin+step_h, Xmin, Xmin+step_w]
    sub_prob = hm[Ymin:Ymin+step_h, Xmin:Xmin+step_w]
    sub_anchor=[]

    iter_h = Ymax-Ymin-step_h
    iter_w = Xmax-Xmin-step_w
    for i in range(iter_h):
        ymin= i+Ymin
        ymax= ymin+step_h
        for j in range(iter_w):
            xmin = j+Xmin
            xmax = xmin+step_w
            
            sub_hm_tmp = hm[ymin:ymax, xmin:xmax]
            val = np.sum(sub_hm_tmp)
            # print(f'(i, j)=({i},{j}), [{ymin},{ymax},{xmin},{xmax}], max_val={max_val}')
            if val>=max_val:
                max_val = val
                bbox[0]= ymin
                bbox[1]= ymax
                bbox[2]= xmin
                bbox[3]= xmax

    sub_prob=hm[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    for i in range(bbox[0], bbox[1]):
        for j in range(bbox[2], bbox[3]):
            sub_anchor.append([j, i])
            # sub_prob.append(hm[j, i])

    # pdb.set_trace()

    return sub_prob, sub_anchor

def anchor_map(anchor, X, Y):
    # import pdb 
    # pdb.set_trace()
    
    result=[ [ X[pt[0]], Y[pt[1]] ] for pt in anchor ]
    return result

def decode_dis(prob, sigma):
    dis=[]
    for p0 in prob:
        for p in p0:
            p = np.clip(p, 0.000000001, 1.0)
            dis_decode = np.sqrt( -2. * sigma * sigma * np.log(p) )
            # dis_decode =  -2. * sigma * sigma * np.log(p)
            dis.append(dis_decode)

    return dis

# linear least squares
def LLS_solve(anchor, dis):
    # import pdb
    # pdb.set_trace()

    length = len(anchor)-1
    X = np.zeros((length, 2), dtype = np.float32)
    Y = np.zeros((length, 1), dtype = np.float32)

    for idx in range(length):
        X[idx][0] = anchor[length][0] - anchor[idx][0]
        X[idx][1] = anchor[length][1] - anchor[idx][1]
        Y[idx][0] = (dis[idx]**2 + anchor[length][0]**2 + anchor[length][1]**2 - dis[length]**2 - anchor[idx][0]**2 - anchor[idx][1]**2) * 0.5
    
    try:
        ret = np.linalg.inv( np.dot(X.transpose(), X) )
    except:
        tmp=np.dot(X.transpose(), X)  
        print(f'anchor={anchor}')
        print(f'X={X}')
        print(f'Y={Y}')
        print(f'tmp={tmp}')
        print(f'tmp矩阵不存在逆矩阵')

    ret = np.dot(ret, X.transpose() )
    ret = np.dot(ret, Y)

    return [ret[0][0], ret[1][0]]
##------------------------------------------

def get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals

def fit_gauss_newton(station, anchor, dis):
    iter_max=20
    cost=0
    last_cost=0
    
    x0=station[0]
    y0=station[1]
    
    import pdb
    # pdb.set_trace()

    std = np.var(dis)
    sigma = np.sqrt(std)
    inv_sigma = 1.0/(sigma+0.000000001)
    inv_sigma=1

    for idx in range(iter_max):
        H = np.zeros( (2,2), dtype = np.float32)
        B = np.zeros( (2,1), dtype = np.float32)
        cost=0
        for idx2 in range(len(anchor)):
            error = (anchor[idx2][0]-x0)**2 + (anchor[idx2][1]-y0)**2 - dis[idx2]**2
            J = np.zeros( (2,1), dtype = np.float32)
            J[0] = -2*(anchor[idx2][0]-x0)
            J[1] = -2*(anchor[idx2][1]-y0)
            H += inv_sigma * inv_sigma * J * J.transpose()
            B += -inv_sigma * inv_sigma * error * J
            cost += error*error
            # print(f'------idx={idx}, idx2={idx2}, cost={cost}, error={error}')
        
        # pdb.set_trace()
        
        delta = np.linalg.solve(H, B)
        if delta[0] is np.nan:
            print(f'*********error in fit gauss newton*********\n')
            break

        if idx > 0 and cost >= last_cost:
            break
        
        x0 += delta[0]
        y0 += delta[1]
        last_cost = cost
    return [x0, y0]


# # #-------------2023-04-23------------
# # #linear least squares
def keypoints_from_heatmaps_tril(heatmaps, preds, center, scale, sigma=2, phase=False):
    if phase:
        preds, maxvals = get_max_preds(heatmaps)

    N, K, H, W = heatmaps.shape
    x = np.arange(0, W, 1, np.float32)
    y = np.arange(0, H, 1, np.float32)
    for n in range(N):
        for k in range(K):
            heatmap = heatmaps[n][k]
            sub_prob, sub_anchor = block_search(heatmap, preds[n][k], H, W, step_h=2, step_w=2, gl=False )
            sub_dis = decode_dis(sub_prob, sigma=sigma)
            sub_anchor = anchor_map(sub_anchor, x, y)
            station = LLS_solve(sub_anchor, sub_dis)

            preds[n][k][0]=station[0]
            preds[n][k][1]=station[1]
    
    return preds

# # #-------------2023-04-23------------
# # # for test
def keypoints_from_heatmaps_tril_img(img, heatmaps, preds, center, scale, sigma=2, phase=False):
    import pdb
    # pdb.set_trace()

    if phase:
        preds, maxvals = get_max_preds(heatmaps)
    
    N, K, H, W = heatmaps.shape
    x = np.arange(0, W, 1, np.float32)
    y = np.arange(0, H, 1, np.float32)

    anchor_16=[[0,0],[0,0],[0,0],[0,0]]
    dis_16=[0,0,0,0]
    pt_16=[0,0]
    scale=4
    scale=32
    for n in range(N):
        for k in range(K):
            heatmap = heatmaps[n][k]
            sub_prob, sub_anchor = block_search(heatmap, preds[n][k], H, W, step_h=2, step_w=2, gl=False )
            sub_dis = decode_dis(sub_prob, sigma=sigma)
            sub_anchor = anchor_map(sub_anchor, x, y)
            station = LLS_solve(sub_anchor, sub_dis)

            # preds[n][k][0]=station[0]
            # preds[n][k][1]=station[1]
            preds[n][k][0]=station[0]*scale
            preds[n][k][1]=station[1]*scale

            if k==16:
                anchor_16=sub_anchor
                dis_16=sub_dis
                pt_16 = preds[n][k]
    
    # for i in range(N):
    #     preds[i] = transform_preds(
    #         preds[i], center[i], scale[i], [W, H], use_udp=False)

    length=len(dis_16)
    # pdb.set_trace()
    for idx in range(length):
        x=int(anchor_16[idx][0]*scale)
        y=int(anchor_16[idx][1]*scale)

        r = int(dis_16[idx]*scale)
        cv2.circle(img, (x, y), r, (0,128,255), 3 )
        cv2.circle(img, (x, y), 3, (255,0,0), -1)

    x=int(pt_16[0])
    y=int(pt_16[1])
    cv2.circle(img, (x, y), 3, (0,0,255), -1)

    return preds, img

# # #-------------2023-04-23------------
# # #------------gauss_newton-iteration--------
def keypoints_from_heatmaps_tril_topK(heatmaps, preds, center, scale, sigma=2, phase=False):
    if phase:
        preds, maxvals = get_max_preds(heatmaps)
    
    N, K, H, W = heatmaps.shape
    x = np.arange(0, W, 1, np.float32)
    y = np.arange(0, H, 1, np.float32)

    anchor=[]
    for i in range(0, H):
        for j in range(0, W):
            anchor.append([j, i])
    anchor = np.array(anchor)
    
    topk=30
    if H==32:
        topk=30
    elif H==16:
        topk=30
    elif H==8:
        topk=15
    elif H==4:
        topk=10

    
    sub_prob, sub_anchor = topK( heatmaps, anchor, topK=[0,topk] )

    for n in range(N):
        for k in range(K):
            sub_dis = decode_dis([sub_prob[n][k]], sigma=sigma)
            station = fit_gauss_newton(preds[n][k], sub_anchor[n][k], sub_dis) #---------------
            
            preds[n][k][0]=station[0]
            preds[n][k][1]=station[1]

    return preds