CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
    configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/cofw/hrnetv2_w18_cofw_256x256_dark.py \
    exp/exp_v2.8.4.2/best_NME_epoch_75.pth \
    4


# python tools/test.py \
#     configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/hrnetv2_w18_wflw_256x256_dark.py \
#     exp/exp_v1.3.0/best_NME_epoch_60.pth \
#     --work-dir  ./exp/exp888/show/ \
#     --eval 'NME' \
#     --out ./exp/exp888/show/out.json \
