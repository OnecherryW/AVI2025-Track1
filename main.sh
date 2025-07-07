nohup python -u mian.py --loss_plot_path --output_model ./output/best_model.pth \
						--training_time 2025-06-30 \
						--train_csv ./data/train_data.csv \
						--val_csv ./data/val_data.csv \
						--test_csv ./data/test_data_basic_information.csv \
						--question q1 q2 q3 q4 q5 q6 \
						--label_col Integrity Collegiality Social_versatility Development_orientation Hireability \
						--rating_csv ./data/all_data.csv \
						--video_dim 1152 \
						--video_dir /home/gdp/AVI/data/face_embedding/siglip2_all_maxP_face \
						--audio_dim 768 \
						--audio_dir /home/gdp/AVI/data/audioFeatures/audioFeatures/emo2vec/max_pooling/emotion2vec_plus_seed \
						--text_dim 4096 \
						--text_dir /home/gdp/AVI/data/text_feature/SFR-Embedding-Mistral \
						--batch_size 64 \
						--learning_rate 1e-4 \
						--num_epochs 200 \
						--log_dir ./log \
						> ./output/20250630_maxpooling.log 2>&1 &
