
# sh tools/dist_test.sh \
#     configs/face/2d_kpt_sview_rgb_img/deeppose/wflw/res50_wflw_256x256_wingloss.py \
#     checkpoints/deeppose_res50_wflw_256x256_wingloss-f82a5e53_20210303.pth \
#     4 \

# sh tools/dist_train.sh \
#     configs/face/2d_kpt_sview_rgb_img/deeppose/wflw/res50_wflw_256x256_wingloss.py \
#     4 \

# python demo/top_down_img_demo.py \
#     configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py \
#     https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
#     --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
#     --out-img-root vis_results

# python demo/top_down_img_demo_with_mmdet.py \
#     demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
#     https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
#     configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py \
#     https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
#     --img-root tests/data/coco/ \
#     --img 000000196141.jpg \
#     --out-img-root vis_results

# configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/hrnetv2_w18_aflw_256x256.py \
# https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \

# python demo/face_img_demo.py \
#     configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/hrnetv2_w18_wflw_256x256_dark.py \
#     https://download.openmmlab.com/mmpose/face/darkpose/hrnetv2_w18_wflw_256x256_dark-3f8e0c2c_20210125.pth \
#     --img-root ./ttt/ \
#     --img 1.png \
#     --out-img-root exp/exp888/ \
#     --radius 1 \

# python demo/face_img_demo.py \
#     configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/cofw/hrnetv2_w18_cofw_256x256_dark.py \
#     exp/exp_v1.12.3/best_NME_epoch_60.pth \
#     --img-root ./ttt/ \
#     --img 1.png \
#     --out-i







# CUDA_VISIBLE_DEVICES=0,1,2 sh tools/dist_train.sh \
#     configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_face/hrnetv2_w18_coco_wholebody_face_256x256_dark.py \
#     3 \
#     --work-dir exp/exp889


# CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py \
#     configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/cofw/hrnetv2_w18_cofw_256x256.py \
#     --work-dir exp/exp888



python tools/test.py \
    configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_face/hrnetv2_w18_coco_wholebody_face_256x256.py \
    exp/exp_v1.5.0/best_NME_epoch_230.pth \
    --work-dir  ./exp/exp888/show/ \
    --eval 'NME' \
    --out ./exp/exp888/show/out.json \


# CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
#     configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/cofw/hrnetv2_w18_cofw_256x256_dark.py \
#     exp/exp_v1.12.3/best_NME_epoch_60.pth \
#     4 

# CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
#     configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/cofw/hrnetv2_w18_cofw_256x256.py \
#     exp/exp_v1.11.3/best_NME_epoch_100.pth \
#     4 
