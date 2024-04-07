load('All_P_LGBM_merge_grid_final.mat')

load('Pt9_10_11_MI_score_plot.mat')
figure (1),
% imagesc(MAUC);axis xy
subplot(2,3,1);imagesc(P1_mergefsore);axis xy;colorbar
subplot(2,3,2);imagesc(P2_mergefsore);axis xy;colorbar
subplot(2,3,3);imagesc(P3_mergefsore);axis xy;colorbar
subplot(2,3,4);imagesc(P4_mergefsore);axis xy;colorbar
subplot(2,3,5);imagesc(P5_mergefsore);axis xy;colorbar
subplot(2,3,6);imagesc(P6_mergefsore);axis xy;colorbar



% subplot(2,3,3);imagesc(P7_mergefsore);axis xy;colorbar
% subplot(2,2,4);imagesc(P8_mergefsore);axis xy;colorbar

figure (2),
% imagesc(MAUC);axis xy
subplot(2,3,1);imagesc(P7_mergefsore);axis xy;colorbar
subplot(2,3,2);imagesc(P8_mergefsore);axis xy;colorbar
subplot(2,3,3);imagesc(P9_mergefsore);axis xy;colorbar
subplot(2,3,4);imagesc(P10_mergefsore);axis xy;colorbar
subplot(2,3,5);imagesc(P11_mergefsore);axis xy;colorbar
% subplot(2,3,6);imagesc(P6_mergefsore);axis xy;colorbar


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% load('Pt9_10_11_MI_score_plot.mat')
% figure (3),
% subplot(2,2,1);imagesc(P9_mergefsore);axis xy;colorbar
% subplot(2,2,2);imagesc(P10_mergefsore);axis xy;colorbar
% subplot(2,2,3);imagesc(P10_mergefsore);axis xy;colorbar
% %subplot(2,2,4);imagesc(P8_mergefsore);axis xy;colorbar