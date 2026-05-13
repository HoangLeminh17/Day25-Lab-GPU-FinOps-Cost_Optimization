# GPU FinOps Lab Report

**Sinh vien:** Le Minh Hoang  
**MSSV:** 2A202600101  
**Lab:** Day 25 - GPU FinOps & Cost Optimization

## 1. Gioi thieu

Muc tieu cua bai lab la xay dung mot quy trinh GPU FinOps hoan chinh gom quan sat tai nguyen, theo doi chi phi, toi uu su dung GPU, danh gia spot instance, autoscaling va phan tich chi phi cho workload thuc te tren GPU. Bai lab giup ket noi giua van hanh he thong va bai toan tai chinh, tu do dua ra cac quyet dinh toi uu hieu nang theo chi phi.

Tong quan, GPU FinOps tap trung vao 3 cau hoi chinh:

- Dang dung bao nhieu tai nguyen GPU?
- Dang ton bao nhieu chi phi va phat sinh lang phi o dau?
- Nen ap dung chien luoc nao de giam chi phi ma van dam bao hieu nang va deadline?

## 2. Phan tich tung phan

### Part 1-7: Mock cluster monitoring va chi phi

#### 2.1. Cluster monitoring

Notebook ghi nhan cum gom **4 nodes** va **8 GPUs** thuoc cac loai **T4, A100, V100**. O thoi diem dau, tat ca GPU deu o trang thai idle, muc su dung trung binh chi **6.8%**, tong bo nho dang dung **12.3/288 GB**, tong cong suat tieu thu **310W**. Dieu nay cho thay cum dang duoc cap phat nhieu tai nguyen hon nhu cau thuc te.

Nhan xet:

- Tai nguyen duoc provision du phong kha nhieu.
- Tai thoi diem baseline, idle GPU la nguon gay lang phi chinh.
- Cum da da dang loai GPU, phu hop cho viec so sanh chi phi va kha nang dat workload.

#### 2.2. Workload submission va billing

He thong submit thanh cong 4 workloads:

- `train-resnet-001` tren T4
- `train-bert-002` tren A100
- `inference-api-003` tren T4
- `train-llm-004` tren 2 GPU A100/V100

Sau khi submit, so GPU ban tang len **5/8**, muc utilization dat **53.1%**. Ve billing:

- Tong chi phi ghi nhan: **$1.1949**
- Tong savings: **$1.2927**
- Muc dung ngan sach: **1.2%**
- Trang thai canh bao: **OK**

Nhan xet:

- He thong ghi nhan billing theo workload ro rang.
- Spot billing tao ra savings dang ke cho cac workload phu hop.
- Muc su dung ngan sach thap, cho thay monitoring va alert hoat dong on dinh.

#### 2.3. Spot instance savings analysis

Bang gia spot cho thay:

- **T4:** $0.2551/h, giam **27.1%** so voi on-demand
- **A100:** $2.0981/h, giam **42.8%**
- **V100:** $1.8621/h, giam **24.9%**

Tat ca 3 yeu cau spot deu duoc cap phat thanh cong. Su kien preemption mo phong khong lam mat instance nao, va bao cao savings cho thay:

- Spot cost: **$0.0013**
- On-demand equivalent: **$0.0042**
- Tong tiet kiem: **$0.0029 (70.0%)**

Nhan xet:

- Spot la chien luoc toi uu rat tot voi workload fault-tolerant.
- Muc savings thuc te rat lon so voi chi phi on-demand.
- Rui ro preemption can duoc can nhac khi workload nhay cam ve thoi gian.

#### 2.4. Autoscaling behavior

Chinh sach autoscaling ban dau co:

- `scale_up_threshold = 80`
- `scale_down_threshold = 20`
- `cooldown_seconds = 60`

Sau khi cap nhat:

- `scale_up_threshold = 70`
- `scale_down_threshold = 25`
- `cooldown_seconds = 30`
- `min_nodes = 2`, `max_nodes = 10`

He thong danh gia 5 chu ky lien tiep va deu tra ve `no_action` vi utilization **53.1%** nam trong khoang **25%-70%**. Trong full workflow, khi utilization tang len **79.9%**, autoscaler dua ra quyet dinh **scale_up**.

Nhan xet:

- Rule autoscaling da phan ung dung theo nguong.
- Khi tai trung binh, he thong tranh scale khong can thiet.
- Khi workload nang duoc dua vao, autoscaler bat dau de xuat mo rong node.

#### 2.5. Waste analysis va recommendations

Trong 5 snapshots lien tiep:

- Tong cost moi interval: **$0.038056**
- Idle cost moi interval: **$0.008833**
- Waste trung binh: **23.2%**

Bao cao waste cho thay:

- Tong chi phi quan sat: **$0.190280**
- Tong idle cost: **$0.044165**
- Tiet kiem theo thang neu toi uu: **$2289.51**
- Muc do: **LOW**

Hai khuyen nghi chinh:

- `USE_SPOT` - tiet kiem uoc tinh **65%**
- `SCHEDULING` - tiet kiem uoc tinh **20%**

Nhan xet:

- Lang phi hien tai chua o muc nguy hiem, nhung van du de tao savings dang ke neu mo rong theo quy mo thang.
- Spot va scheduling la hai quick wins de giam chi phi nhanh.

#### 2.6. Full FinOps workflow

Part 7 tong hop toan bo quy trinh:

- Baseline: **8 GPUs**, utilization **53.1%**
- Sau khi them workload nang: utilization **79.9%**, busy **8/8**
- Autoscaler quyet dinh: **scale_up**
- Waste giam con **4.9%**
- Spot savings bo sung: **$0.1814 (70.0%)**
- Tong spend cuoi: **$1.3640**
- Tong savings cuoi: **$1.4151**

Ket luan cho Part 1-7:

- Cluster monitoring giup nhin thay ngay idle capacity.
- Billing va snapshots tao du lieu de luong hoa lang phi.
- Spot va autoscaling la hai don bay quan trong de toi uu chi phi.

### Part 8: Real GPU training tren Kaggle/Colab

#### 2.7. Real GPU detection va metrics

Notebook detect thanh cong **Tesla T4**, dung luong **15.6 GB**, CUDA **12.8**, gia on-demand **$0.35/h**. Diagnostic thong bao `pynvml available = True`, cho phep doc:

- GPU util
- Memory usage
- Power draw
- Temperature

Dieu nay xac nhan moi truong thuc nghiem du dieu kien de do hieu nang va chi phi tren GPU thuc.

#### 2.8. FP32 vs Mixed Precision (AMP)

Ket qua train ResNet-18 tren CIFAR-10:

**FP32**

- Tong thoi gian: **118.5s**
- Peak memory: **0.82 GB**
- Avg GPU util: **94.3%**
- Avg power: **67.8W**
- Estimated cost: **$0.011523**

**AMP**

- Tong thoi gian: **57.3s**
- Peak memory: **0.60 GB**
- Avg GPU util: **89.0%**
- Avg power: **65.7W**
- Estimated cost: **$0.005570**

**Cai thien khi dung AMP**

- Nhanh hon **2.07x**
- Giam **0.22 GB** peak memory
- Tiet kiem **$0.005953**
- Giam **51.7%** chi phi

Savings khi scale:

- 1 ngay training: tiet kiem **$4.34**
- 1 tuan training: tiet kiem **$30.38**
- 1 thang training: tiet kiem **$130.18**

Nhan xet:

- AMP la ky thuat co hieu qua rat cao va rui ro thap.
- Loi ich ro nhat la giam thoi gian train va chi phi ma van giu chat luong huan luyen chap nhan duoc.
- Day la chien luoc nen uu tien dau tien trong toi uu chi phi GPU.

#### 2.9. Reporting real GPU costs

Notebook da day chi phi thuc te len gateway:

- FP32 workload: **$0.011500**
- AMP workload (spot): **$0.001700**
- Savings ghi nhan: **$0.003900**

Dashboard cuoi sau khi cong gop voi mock platform:

- Total platform cost: **$1.3640**
- Total savings: **$1.4151**
- Budget utilization: **1.4%**
- Alert: **OK**

Ket luan cho Part 8:

- Real GPU telemetry giup bai lab chuyen tu mo phong sang thong so thuc.
- AMP tao ra gia tri FinOps ro rang nhat trong phan thuc nghiem.

### Part 8.5: Advanced GPU cost optimization

#### 2.10. Multi-GPU scaling efficiency

Tu baseline `1x T4`, bang phan tich cho thay:

- **1 GPU:** cost thap nhat **$0.0115**
- **8 GPU:** runtime nhanh nhat **0.0057h**
- **8 GPU:** cost/performance tot nhat, chi **$0.0027** moi speedup unit

Nhan xet:

- Tang so GPU khong phai luc nao cung toi uu chi phi tuyet doi.
- Neu uu tien thoi gian va cost/performance, 8 GPU la phuong an hop ly hon.

#### 2.11. Project cost forecasting

Du bao project gom 4 phase:

- Data Preparation
- Model Training
- Hyperparameter Tuning
- Model Evaluation

Tong hop forecast:

- Base cost: **$3551.20**
- Contingency 20%: **$710.24**
- Expected total: **$4261.44**
- Best case: **$2578.82**
- Worst case: **$5233.82**
- 95% confidence interval: **$2913.09 - $5609.79**

Nhan xet:

- Giai doan training va tuning chiem phan lon nhat trong tong chi phi.
- Them contingency la can thiet vi uncertainty cua phase A100 rat cao.

#### 2.12. Optimization strategy prioritization

Bang uu tien chien luoc cho baseline **$1468.00**:

- Switch to Mixed Precision (AMP): save **$367.0**
- Use Spot Instances: save **$880.8**
- Optimize Batch Size: save **$220.2**
- Switch to More Efficient GPU Type: save **$587.2**
- Implement Early Stopping: save **$293.6**

Ket qua roadmap:

- Max cumulative savings: **$1288.32**
- Optimized cost: **$179.68**
- Top priority: **Switch to Mixed Precision (AMP)**

Nhan xet:

- AMP la quick win tot nhat vi effort thap, risk thap.
- Spot cho savings lon nhung risk cao hon.
- Chien luoc nen duoc trien khai theo lo trinh thay vi lam dong loat.

#### 2.13. Challenge strategy

Trong bai toan fine-tuning LLM:

- Baseline: **8x A100 trong 200h**
- Baseline cost: **$5872.00**
- Deadline: **2 tuan**
- Budget: **$5000**

Phuong an duoc chon:

- Giu cau hinh **8x A100** de dat deadline
- Uu tien 3 chien luoc:
  - Mixed Precision (AMP)
  - Optimize Batch Size
  - Switch to More Efficient GPU Type

Ket qua:

- Forecast final cost: **$3167.09**
- 95% CI sau savings: **$865.26 - $5468.92**
- Budget check: **PASS**
- Headroom con lai: **$1832.91**

Nhan xet:

- Chien luoc toi uu da dua tong chi phi ve duoi budget trong khi van giu constraint ve deadline.
- Day la minh hoa ro cho cach FinOps ket hop performance engineering, forecasting va risk control.

## 3. Ket luan va hoc hoi

Sau bai lab nay, em rut ra cac bai hoc chinh:

- Monitoring la buoc dau tien de toi uu chi phi, vi khong do thi khong the toi uu.
- Idle GPU va overprovisioning la nguon lang phi pho bien trong he thong AI/ML.
- Spot instance phu hop cho workload fault-tolerant va co the mang lai savings rat lon.
- Mixed Precision (AMP) la ky thuat toi uu chi phi co hieu qua cao, de ap dung va it rui ro.
- Autoscaling giup can bang giua hieu nang va chi phi khi tai thay doi theo workload.
- Forecasting va confidence interval can thiet khi quyet dinh tren du an GPU co quy mo lon.

Ve ung dung thuc te, cac chien luoc nay co the ap dung truc tiep trong:

- Training pipeline cho computer vision, NLP, LLM fine-tuning
- GPU inference cluster co tai thay doi theo gio
- Mo hinh MLOps/Platform engineering can kiem soat budget cloud

## 4. Tep dinh kem

Bao cao nay di kem voi:

- Thu muc `screenshots/` gom day du screenshot theo tung part
- Thu muc `generated_charts/` gom 9 file PNG visualization
- Notebook `notebook/gpu_finops_lab.ipynb` da giu nguyen outputs

## 5. Ket luan chung

Bai lab da mo phong va thuc hien thanh cong mot quy trinh GPU FinOps tu co ban den nang cao. Ket qua quan trong nhat la chung minh duoc rang cac ky thuat nhu mixed precision, spot usage, autoscaling, forecasting va optimization roadmap co the giam chi phi dang ke nhung van bao toan muc tieu van hanh. Day la nen tang quan trong cho viec van hanh he thong AI hieu qua va ben vung ve chi phi.
