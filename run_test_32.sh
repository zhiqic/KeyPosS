a=0


if [ $a == 1 ]
then

    ######wflw
    printf "********************** NME of heatmap on wflw ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/hrnetv2_w18_wflw_256x256.py \
        exp/exp_v1.0.9.2/best_NME_epoch_230.pth \
        4 
    printf "********************** end heatmap on wflw ****************************\n"
    printf "\n"
    
    
    ######coco
    printf "********************** NME of heatmap on coco ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_face/hrnetv2_w18_coco_wholebody_face_256x256.py \
        exp/exp_v1.5.1/best_NME_epoch_380.pth \
        4 
    printf "********************** end heatmap on coco ****************************\n"
    printf "\n"
    
    
    ######300w
    printf "********************** NME of heatmap on 300w ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/300w/hrnetv2_w18_300w_256x256.py \
        exp/exp_v1.7.1/best_NME_epoch_90.pth \
        4 
    printf "********************** end heatmap on 300w ****************************\n"
    printf "\n"
    
    
    ######aflw
    printf "********************** NME of heatmap on aflw ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/hrnetv2_w18_aflw_256x256.py \
        exp/exp_v1.9.1/best_NME_epoch_60.pth \
        4 
    printf "********************** end heatmap on aflw ****************************\n"
    printf "\n"
    
    
    ######cofw
    printf "********************** NME of heatmap on cofw ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/cofw/hrnetv2_w18_cofw_256x256.py \
        exp/exp_v1.11.1/best_NME_epoch_90.pth \
        4 
    printf "********************** end heatmap on cofw ****************************\n"
    printf "\n"

else  ##-----------------------------------------dark-pose

    ######wflw
    printf "********************** NME of dark pose on wflw ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/hrnetv2_w18_wflw_256x256_dark.py \
        exp/exp_v1.3.1/best_NME_epoch_230.pth \
        4 
    printf "********************** end darkpose on wflw ****************************\n"
    printf "\n"
    
    
    ######coco
    printf "********************** NME of dark pose on coco ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/coco_wholebody_face/hrnetv2_w18_coco_wholebody_face_256x256_dark.py \
        exp/exp_v1.4.1/best_NME_epoch_160.pth \
        4 
    printf "********************** end darkpose on coco ****************************\n"
    printf "\n"
    
    ######300w
    printf "********************** NME of dark pose on 300w ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/300w/hrnetv2_w18_300w_256x256_dark.py \
        exp/exp_v1.8.1/best_NME_epoch_95.pth \
        4 
    printf "********************** end darkpose on 300w ****************************\n"
    printf "\n"
    
    
    ######aflw
    printf "********************** NME of dark pose on aflw ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/hrnetv2_w18_aflw_256x256_dark.py \
        exp/exp_v1.10.1/best_NME_epoch_100.pth \
        4 
    printf "********************** end darkpose on aflw ****************************\n"
    printf "\n"
    
    
    ######cofw
    printf "********************** NME of dark pose on cofw ****************************\n"
    CUDA_VISIBLE_DEVICES=0,1,2,3 sh tools/dist_test.sh \
        configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/cofw/hrnetv2_w18_cofw_256x256_dark.py \
        exp/exp_v1.12.1/best_NME_epoch_100.pth \
        4 
    printf "********************** end darkpose on cofw ****************************\n"
    printf "\n"
fi