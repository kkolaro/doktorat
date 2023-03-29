
# ANALIZA PERFORMANSI KLASIFIKACIONIH ALGORITAMA


library(imbalance)
library(rcompanion)
library(rattle)
library(DescTools)
library(mgcv)
library(vip)
library(foreign)
library(htmlTable)
library(ltm)
library(mice)
library(mda)# fda regresiju
library(randomForest)
library(caret)#train
library(RWeka)
library(psych)
library(lares)
library(xgboost)
library(tibble)
library(Amelia)
library(rpart)
library(factoextra)
library(FactoMineR)
library(factoextra)
library(devtools)
library(corrplot)
library(ROCR)
library(pROC)
library(magrittr)
library(dplyr)# %>%
library(Matrix)
library(class)
library(tidyr)
library(leaps)
library(glmnet)
library(e1071) # svm
library(rpart.plot)
library(psych)
library(stats)
library(MASS)
library(AICcmodavg)
library(car)
library(ROSE)
library(h2o)


# 1. FORMIRANJE  DATA SETA i transformacija

# Import podataka

default_cr<-read.csv(file="C:\\Users\\kkolaro\\OneDrive - ENDAVA\\Desktop\\Doktorat\\Rad\\last\\default_cr.csv", stringsAsFactors = TRUE)
default_cr_norm<-read.csv(file="C:\\Users\\kkolaro\\OneDrive - ENDAVA\\Desktop\\Doktorat\\Rad\\last\\default_cr_norm.csv", stringsAsFactors = TRUE)
default_cr_h2o<-read.csv(file="C:\\Users\\kkolaro\\OneDrive - ENDAVA\\Desktop\\Doktorat\\Rad\\last\\default_cr_h2o.csv", stringsAsFactors = FALSE)
default_pc<-read.csv(file="C:\\Users\\kkolaro\\OneDrive - ENDAVA\\Desktop\\Doktorat\\Rad\\last\\default_pc.csv", stringsAsFactors = TRUE)

# Faktor zavisna promenjiva

default_cr$default.payment.next.month <- as.factor(default_cr$default.payment.next.month)
levels(default_cr$default.payment.next.month)

default_cr_norm$default.payment.next.month <- as.factor(default_cr_norm$default.payment.next.month)
levels(default_cr_norm$default.payment.next.month)

default_cr_h2o$default.payment.next.month <- as.factor(default_cr_h2o$default.payment.next.month)
levels(default_cr_h2o$default.payment.next.month)

default_pc$default.payment.next.month <- as.factor(default_pc$default.payment.next.month)
levels(default_pc$default.payment.next.month)


# Kreiranje trening i testing podataka

split <- createDataPartition (default_cr$default.payment.next.month, p = .7, list = F)# caret

train <- default_cr[split,]
test  <- default_cr[-split,]

train_norm<-default_cr_norm[split,]
test_norm<-default_cr_norm[-split,]

train_pc<-default_pc[split,]
test_pc<-default_pc[-split,]


actuals<-test$default.payment.next.month# sacuvacemo originalne vrednosti za K NN i SVM, potrebne usled naknadnih transformacija u ovim algoritmima

# Random over sampling (Menardi, Torelli, 2014)

train<-ROSE(default.payment.next.month ~ ., data  = train)$data  
table(train$default.payment.next.month)# broj uzoraka ostaje isti

train_norm<-ROSE(default.payment.next.month ~ ., data  = train_norm)$data  
table(train_norm$default.payment.next.month)# broj uzoraka ostaje isti


train_pc<-ROSE(default.payment.next.month ~ ., data  = train_pc)$data  
table(train_pc$default.payment.next.month)# broj uzoraka ostaje isti


# 2. PREDIKTIVNI MODELI

# 2.1 LOGISTICKA REGRESIJA

# Kreiranje modela

model_lr<-glm(default.payment.next.month ~ ., data=train_norm, family = binomial)
summary(model_lr)

# Najvazniji, najinformativniji prediktori

vip(model_lr,num_features = 10, method= "model")

# Predikcija

predict_lr<-model_lr%>%predict(test_norm,type="response")

# Brier Score test, od 0 do 1. sto blize 0, manja vrednost, bolji model

bs_lr<-round(BrierScore(as.numeric(test_norm$default.payment.next.month)-1,predict_lr),3)

# AUC

roc_lr<-roc(test_norm$default.payment.next.month,predict_lr)
lr_auc<-round(as.numeric(roc_lr$auc),4)
th<-coords(roc_lr, "best", ret="threshold")$threshold  #pROC


# Matrica konfuzije

predicted_lr<-ifelse(predict_lr>th,"1","0")
cm_lr<-confusionMatrix(factor(predicted_lr), test_norm$default.payment.next.month, positive = '1',mode = "everything")


#2.1.1 Log reg sa PC

model_lrpc<-glm(default.payment.next.month ~ ., data=train_pc, family = binomial)#model sa PC
summary(model_lrpc)

# Predikcija

predict_lrpc<-model_lrpc%>%predict(test_pc,type="response")# sa PC

# Brier Score test, od 0 do 1. sto blize 0, manja vrednost, bolji model

bs_lrpc<-round(BrierScore(as.numeric(test_pc$default.payment.next.month)-1,predict_lrpc),3)

# AUC

roc_lrpc<-roc(test_pc$default.payment.next.month,predict_lrpc)
lrpc_auc<-round(as.numeric(roc_lrpc$auc),3)
th<-coords(roc_lrpc, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_lrpc<-ifelse(predict_lrpc>th,"1","0")
cm_lrpc<-confusionMatrix(factor(predicted_lrpc), test_pc$default.payment.next.month, positive = '1',mode = "everything")

# 2.2 PENALIZED LOGISTIC REGRESSION, regularizacija linearnih modela 

#Lasso  regresija

# Kreiranje matrice prediktora 

x_train<- model.matrix(default.payment.next.month ~., data=train_norm,)[,-24]
x_test<-model.matrix(default.payment.next.month ~., test_norm,)[,-24]

y_output<-train_norm$default.payment.next.month

# Koriscenjem CV trazimo najoptimalniju vrednost lambda, kojom se odredjuje stepen 'sankcionisanja' kompleksnosti modela tj. umanjenje reg koeficijenata 

lasso_param<-cv.glmnet(x_train,y_output,alpha=1,family="binomial")# glmnet package.  Za alpha 0=1, radi se o Lasso reg

# Kreiranje modela

model_lrlasso<- glmnet(x_train,y_output, alpha = 1, family = "binomial",lambda = lasso_param$lambda.min)
model_lrlasso$beta

# Predikcija

predict_lasso<-model_lrlasso%>%predict(newx=x_test,type="response")

# Brier Score test

bs_lasso<-round(BrierScore(as.numeric(test_norm$default.payment.next.month)-1,predict_lasso),3)

# AUC

roc_lasso<-roc(test_norm$default.payment.next.month,as.numeric(predict_lasso))
lasso_auc<-round(as.numeric(roc_lasso$auc),4)
th<-coords(roc_lasso, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_lasso<-ifelse(predict_lasso>th,"1","0")
cm_lasso<-confusionMatrix(factor(predicted_lasso), test_norm$default.payment.next.month, positive = '1',mode = "everything")

# Najvazniji, najinformativniji prediktori

vip(model_lrlasso,num_features = 10, method= "model")

# Ridge regresija

ridge_param<-cv.glmnet(x_train,y_output,alpha=0,family="binomial")

# Kreiranje modela

model_ridge<- glmnet(x_train,y_output, alpha = 0, family = "binomial",lambda = ridge_param$lambda.min)

# Predikcija

predict_ridge<-model_ridge%>%predict(newx=x_test,type="response")

# Brier score test

bs_ridge<-round(BrierScore(as.numeric(test_norm$default.payment.next.month)-1,predict_ridge),3)

# AUC

roc_ridge<-roc(test_norm$default.payment.next.month,as.numeric(predict_ridge))
ridge_auc<-round(as.numeric(roc_ridge$auc),4)
th<-coords(roc_ridge, "best", ret="threshold")$threshold  #pROC

#Matrica konfuzije

predicted_ridge<-ifelse(predict_ridge>th,"1","0")
cm_ridge<-confusionMatrix(factor(predicted_ridge), test_norm$default.payment.next.month, positive = '1',mode = "everything")

# Najvazniji, najinformativniji prediktori

vip(model_ridge,num_features = 10, method= "model")

# Elastic net regresija

elas<-train(default.payment.next.month ~ ., data=train_norm,method="glmnet",trControl=trainControl("cv",number = 10),tuneLength=10)

elasticnet_param<-cv.glmnet(x_train,y_output,alpha=elas$bestTune$alpha,family="binomial")#Za alpha izmedju 0 i 1, radi se o Elastic Net regresiji

# Kreiranje modela

model_elasnet<- glmnet(x_train,y_output, alpha = elas$bestTune$alpha, family = "binomial",lambda = elasticnet_param$lambda.min)

# Predikcija

predict_elasnet<-model_elasnet%>%predict(newx=x_test,type="response")

# Brier score test

bs_elasnet<-round(BrierScore(as.numeric(test_norm$default.payment.next.month)-1,predict_elasnet),3)

# AUC

roc_elasnet<-roc(test_norm$default.payment.next.month,as.vector(predict_elasnet))
elasnet_auc<-round(as.numeric(roc_elasnet$auc),4)
th<-coords(roc_elasnet, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_elasnet<-ifelse(predict_elasnet>th,"1","0")
cm_elsnet<-confusionMatrix(factor(predicted_elasnet), test_norm$default.payment.next.month, positive = '1',mode = "everything")

# Najvazniji, najinformativniji prediktori

vip(model_elasnet,num_features = 10, method= "model")


# Poredjenje linearni regresioni modelo

model_poredjenje<-data.frame(Model=c("LinReg","LinRegPc","Lasso","Ridge","ElasNet"),
                             
                             Tacnost=c(round(cm_lr$overall[1],3),
                                       round(cm_lrpc$overall[1],3),
                                       round(cm_lasso$overall[1],3),
                                       round(cm_ridge$overall[1],3),
                                       round(cm_elsnet$overall[1],3)),
                             
                             Sensitivity=c(round(cm_lr$byClass[1],3),
                                           round(cm_lrpc$byClass[1],3),
                                           round(cm_lasso$byClass[1],3),
                                           round(cm_ridge$byClass[1],3),
                                           round(cm_elsnet$byClass[1],3)),
                             
                             Specificity=c(round(cm_lr$byClass[2],3),
                                           round(cm_lrpc$byClass[2],3),
                                           round(cm_lasso$byClass[2],3),
                                           round(cm_ridge$byClass[2],3),
                                           round(cm_elsnet$byClass[2],3)),
                             F1 =c(round(cm_lr$byClass[7],3),
                                   round(cm_lrpc$byClass[7],3),
                                   round(cm_lasso$byClass[7],3),
                                   round(cm_ridge$byClass[7],3),
                                   round(cm_elsnet$byClass[7],3)),
                             
                             Kappa=c(round(cohen.kappa(cm_lr$table)$kappa,3),
                                     round(cohen.kappa(cm_lrpc$table)$kappa,3),
                                     round(cohen.kappa(cm_lasso$table)$kappa,3),
                                     round(cohen.kappa(cm_ridge$table)$kappa,3),
                                     round(cohen.kappa(cm_elsnet$table)$kappa,3)),
                             
                             AUC   =c(lr_auc,lrpc_auc,lasso_auc,ridge_auc,elasnet_auc),
                             
                             BS=c(bs_lr,bs_lrpc, bs_lasso,bs_ridge,bs_elasnet))


model_poredjenje%>%htmlTable(caption="Pokazatelji performansi regresionih modela")   

# AUC krive

roc(test_norm$default.payment.next.month,predict_lr,plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.main=0.8, main="AUC, Lin regresionih modela",
    col="chocolate",identity.col="chocolate",print.thres.col="chocolate",print.thres.pch=10,print.auc.x=.5,print.auc.y=.45)
plot.roc(test_norm$default.payment.next.month,as.vector(predict_lasso),print.auc = T, add = T,lwd= 1,lty=1,col="red",print.auc.x=.5,print.auc.y=.37)
plot.roc(test_norm$default.payment.next.month,as.vector(predict_ridge),print.auc = T, add = T,lwd= 1,lty=1,col="blue",print.auc.x=.5,print.auc.y=.29)
plot.roc(test_norm$default.payment.next.month,as.vector(predict_elasnet),print.auc = T, add = T,lwd= 1,lty=1,col="green",print.auc.x=.5,print.auc.y=.21)
plot.roc(test_pc$default.payment.next.month,as.vector(predict_lrpc),print.auc = T, add = T,lwd= 1,lty=1,col="yellow",print.auc.x=.5,print.auc.y=.13)

legend("topleft", legend=c("LR","Lasso","Ridge","ElasNet", "LRpc"), col=c("chocolate","red","blue","green","yellow"), lwd=1, cex=.8)


# 2.3 LDA i QDA 

# LDA

model_lda<-lda(default.payment.next.month ~ ., data=train_norm)
model_lda

par(mar=c(1,1,1,1))
plot(model_lda)# x osa je LD1

# Predikcija

predict_lda<-model_lda%>%predict(test_norm)
names(predict_lda)
head(predict_lda$posterior,2)

# Brier score test

bs_lda<-round(BrierScore(as.numeric(test_norm$default.payment.next.month)-1,predict_lda$posterior[,2]),3)

# AUC

roc_lda<-roc(test_norm$default.payment.next.month,predict_lda$posterior[,2])
lda_auc<-round(as.numeric(roc_lda$auc),3)
th<-coords(roc_lda, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_lda<-ifelse(predict_lda$posterior[,2] > th,"1","0")
cm_lda<-confusionMatrix(factor(predicted_lda), test_norm$default.payment.next.month, positive = '1')

# Najvazniji, najinformativniji prediktori

coef(model_lda)
df_regkoeficijenti<-data.frame(coef(model_lda))
impattr_lda<-rownames_to_column(df_regkoeficijenti)# package tibble
colnames(impattr_lda)<-c('Prediktori','RegKoeficijenti')
impattr_lda[order(-abs(impattr_lda$RegKoeficijenti)),][c(1:10),]# najveci reg koeficijenata po apsolutnoj vrednosti 

# QDA

model_qda<-qda(default.payment.next.month ~ . -EDUCATION,data=train_norm)# Error in qda.default(x, grouping, ...) : rank deficiency in group 1, sa EDUCATION u modelu

# Predikcija

predict_qda<-model_qda%>%predict(test_norm)

# Brier score test

bs_qda<-round(BrierScore(as.numeric(test_norm$default.payment.next.month)-1,predict_qda$posterior[,2]),3)

# AUC

roc_qda<-roc(test_norm$default.payment.next.month,predict_qda$posterior[,2])
qda_auc<-round(as.numeric(roc_qda$auc),3)
th<-coords(roc_qda, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_qda<-ifelse(predict_qda$posterior[,2] >th,"1","0")
cm_qda<-confusionMatrix(factor(predicted_qda), test_norm$default.payment.next.month, positive = '1')

# Tacnost modela LDA i QDA

model_poredjenje<-data.frame(Model=c("LDA","QDA"),
                             Tacnost=c(round(cm_lda$overall[1],3),
                                       round(cm_qda$overall[1],3)),
                             
                             Sensitivity=c(round(cm_lda$byClass[1],3),
                                           round(cm_qda$byClass[1],3)),
                             
                             Specificity=c(round(cm_lda$byClass[2],3),
                                           round(cm_qda$byClass[2],3)),
                             
                             F1 =c(round(cm_lda$byClass[7],3),
                                   round(cm_qda$byClass[7],3)),
                             
                             Kappa=c(round(cohen.kappa(cm_lda$table)$kappa,3),
                                     round(cohen.kappa(cm_qda$table)$kappa,3)),
                             AUC=   c(lda_auc,qda_auc),
                             BS=   c(bs_lda,bs_qda))

model_poredjenje%>%htmlTable(caption="Pokazatelji performansi LDA i QDA modela")

# AUC krive

roc(test_norm$default.payment.next.month,predict_lda$posterior[,2],plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.main=0.8, main="LDA i QDA AUC",
    col="blue",identity.col="blue",print.thres.col="blue",print.auc.x=.5,print.auc.y=.43)
plot.roc(test_norm$default.payment.next.month,predict_qda$posterior[,2],print.auc = T, add = T,lwd= 1,lty=1,col="red",print.auc.x=.5,print.auc.y=.33)
legend("topleft", legend=c("LDA","QDA"), col=c("blue","red"), lwd=1, cex=.8)

# Statisticki znacaj AUC razlika

roc.test(roc_lda,roc_qda,method=c("delong"))# statisticki znacajna razlika

# 3.1 STABLA ODLUCIVANJA

trControl=trainControl("cv",number=5)

# Kreiranje modela

model_tree<-train(default.payment.next.month~., data=train,method="rpart",trControl= trControl, tuneLength=10)# tuneLength daje moguce cp vrednosti
plot(model_tree)
model_tree$bestTune
fancyRpartPlot(model_tree$finalModel)

# Najinformativniji prediktori

vip(model_tree, num_features = 5, bar=F)

# Predikcija primenom DT

predict_tree<-predict(model_tree,test,type="prob")

# Brier score test

bs_tree<-round(BrierScore(as.numeric(test$default.payment.next.month)-1,predict_tree[,2]),3)

# AUC

roc_tree<-roc(test$default.payment.next.month,predict_tree[,2])
tree_auc<-round(as.numeric(roc_tree$auc),3)
th<-coords(roc_tree, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_tree <- ifelse(predict_tree[,2] > th, 1,0)
cm_tree<-confusionMatrix(factor(predicted_tree),
                         test$default.payment.next.month,positive = "1")

# 3.2 RANDOM FOREST

model_rf<-train(default.payment.next.month~., data=train,method="rf",trControl= trControl,tuneLength=5)# smanjana tunelength na 5, zbog performansi
plot(model_rf)
model_rf$bestTune # mtry br. prediktora za svako stablo
cat("Optimalan broj prediktora koji ce se koristiti pri izradi stabala je" ,model_rf$finalModel$mtry )
model_rf$finalModel 
# AUC krive
# Najinformativniji prediktori

vip(model_rf, num_features = 5, bar=F)

# Predikcija primenom DT

predict_rf<-data.frame(predict(model_rf,test,type="prob"))

# Brier score test

bs_rf<-round(BrierScore(as.numeric(test$default.payment.next.month)-1,predict_rf$X1),3)

# AUC

roc_rf<-roc(test$default.payment.next.month,predict_rf$X1)
rf_auc<-round(as.numeric(roc_rf$auc),3)
th<-coords(roc_rf, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_rf <- ifelse(predict_rf$X1 > th, 1,0)
cm_rf<-confusionMatrix(factor(predicted_rf),
                       test$default.payment.next.month,positive = "1")

# 3.3 BOOSTING MODEL,  xgboost sa xgbTree. Koristi sve corove procesora, paralelno procesiranje


model_xgb<-train(default.payment.next.month~., data=train,method="xgbTree",trControl= trControl)# smanjana tunelength na 5, zbog performansi

model_xgb$bestTune # mtry br. prediktora za svako stablo

# Najinformativniji prediktori

vip(model_xgb, num_features = 5, bar=F)

# Predikcija primenom DT

predict_xgb<-data.frame(predict(model_xgb,test,type="prob"))

# Brier score test

bs_xgb<-round(BrierScore(as.numeric(test$default.payment.next.month)-1,predict_xgb$X1),3)

# AUC

roc_xgb<-roc(test$default.payment.next.month,predict_xgb$X1)
xgb_auc<-round(as.numeric(roc_xgb$auc),3)
th<-coords(roc_xgb, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_xgb <- ifelse(predict_xgb$X1 > th, 1,0)
cm_xgb<-confusionMatrix(factor(predicted_xgb),
                        test$default.payment.next.month,positive = "1")


# Poredjenje modela na bazi stabla

model_poredjenje<-data.frame(Model=c("DT","RF", "GTB"),
                             Tacnost=c(round(cm_tree$overall[1],3),
                                       round(cm_rf$overall[1],3),
                                       round(cm_xgb$overall[1],3)),
                             
                             Sensitivity=c(round(cm_tree$byClass[1],3),
                                           round(cm_rf$byClass[1],3),
                                           round(cm_xgb$byClass[1],3)),
                             
                             Specificity=c(round(cm_tree$byClass[2],3),
                                           round(cm_rf$byClass[2],3),
                                           round(cm_xgb$byClass[2],3)),
                             
                             F1 =        c(round(cm_tree$byClass[7],3),
                                           round(cm_rf$byClass[7],3),
                                           round(cm_xgb$byClass[7],3)),
                             
                             Kappa=       c(round(cohen.kappa(cm_tree$table)$kappa,3),
                                            round(cohen.kappa(cm_rf$table)$kappa,3),
                                            round(cohen.kappa(cm_xgb$table)$kappa,3)),
                             
                             AUC=          c(tree_auc,rf_auc, xgb_auc),
                             
                             BS=          c(bs_tree,bs_rf,bs_xgb))

model_poredjenje%>%htmlTable(caption="Pokazatelji performansi modela na bazi stabla")



# Statisticki znacaj AUC razlika
roc(test$default.payment.next.month,predict_tree[,2],plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.main=0.8, main="AUC, modela na bazi stabla",
    col="chocolate",identity.col="chocolate",print.thres.col="chocolate",print.auc.x=.5,print.auc.y=.49)
plot.roc(test$default.payment.next.month,predict_rf$X1,print.auc = T, add = T,lwd= 1,lty=1,col="red",print.auc.x=.5,print.auc.y=.41)
plot.roc(test$default.payment.next.month,predict_xgb$X1,print.auc = T, add = T,lwd= 1,lty=1,col="blue",print.auc.x=.5,print.auc.y=.32)
legend("topleft", legend=c("DT","RF","GTB"), col=c("chocolate","red","blue"), lwd=1, cex=.8)


roc.test(roc_tree,roc_rf,method=c("delong"))# statisticki znacajna razlika


# 4. NB

# Kreiranje modela

model_nb<-naiveBayes(train$default.payment.next.month~., data=train)
model_nb # vidimo mean i sd za svaki numericki prediktor razlicitih klasa

train%>%filter(default.payment.next.month =="Default")%>%summarise(mean(PAY_AMT1),sd(PAY_AMT2))

# Predikcija

predict_nb<-predict(model_nb,test,type='raw')[,2]

# Brier score test

bs_nb<-round(BrierScore(as.numeric(test$default.payment.next.month)-1,predict_nb),3)

# AUC

roc_nb<-roc(test$default.payment.next.month,predict_nb)
nb_auc<-round(as.numeric(roc_nb$auc),3)
th<-coords(roc_nb, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_nb <- ifelse(predict_nb > th, 1,0)
cm_nb<-confusionMatrix(factor(predicted_nb),
                       test$default.payment.next.month,positive = "1")

roc_nb<-roc(test$default.payment.next.month,predict_nb,plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.main=0.8, main="ROC, NB",
            col="red",identity.col="red",print.thres.col="red",print.auc.x=.5,print.auc.y=.49)

model_poredjenje<-data.frame(Model="NB",Tacnost= round(cm_nb$overall[1],3),
                             
                             Sensitivity=round(cm_nb$byClass[1],3),
                             Specificity=round(cm_nb$byClass[2],3),
                             F1 =round(cm_nb$byClass[7],3),
                             Kappa= round(cohen.kappa(cm_nb$table)$kappa,3),
                             AUC   = round(as.numeric(roc_nb$auc),3),
                             BS= bs_nb)    

model_poredjenje%>%htmlTable(caption="Pokazatelji performansi NB modela")

# 5. KNN model

# Kreiranje modela 

model_knn<-train(default.payment.next.month ~., data = train, method='knn', 
                 trControl=trControl,
                 tuneLength=5)# teuneLength broj mogucih vrednosti  K 

model_knn$bestTune# optimalna vrednost suseda k
plot(model_knn,xlab="Broj suseda K",ylab='Tacnost u funkciji od K', type='b',col='red',lwd=1.5,pch='o')

# Najznacajniji prediktori za tacnost modela

impattr_knn<-data.frame(varImp(model_knn)[[1]])
impattr_knn<-rownames_to_column(impattr_knn)# package tibble
colnames(impattr_knn)<-c('Prediktori','RegKoeficijenti')
impattr_knn[order(-abs(impattr_knn$RegKoeficijenti)),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

#Predikcija

predict_knn<-predict(model_knn, test, type = "prob")[,2] # X1 stecaj

# Brier score test

bs_knn<-round(BrierScore(as.numeric(test$default.payment.next.month)-1,predict_knn),3)

# AUC

roc_knn<-roc(test$default.payment.next.month,predict_knn)
knn_auc<-round(as.numeric(roc_knn$auc),3)
th<-coords(roc_knn, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_knn <- ifelse(predict_knn > th, 1,0)
cm_knn<-confusionMatrix(factor(predicted_knn),
                        factor(actuals) ,positive = "1")

roc_knn<-roc(test$default.payment.next.month,predict_knn,plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.main=0.8, main="ROC, KNN",
             col="red",identity.col="red",print.thres.col="red",print.auc.x=.5,print.auc.y=.49)


model_poredjenje<-data.frame(Model="KN",Tacnost= round(cm_knn$overall[1],3),
                             
                             Sensitivity=round(cm_knn$byClass[1],3),
                             Specificity=round(cm_knn$byClass[2],3),
                             F1 =round(cm_knn$byClass[7],3),
                             Kappa= round(cohen.kappa(cm_knn$table)$kappa,3),
                             AUC   = round(as.numeric(roc_knn$auc),3),
                             BS= bs_knn)  

model_poredjenje%>%htmlTable(caption="Pokazatelji performansi K NN modela")


# 6  SUPORT VECTOR MACHINE linear

#  Kreiranje sintaticki ispravnih imena, za vrednosti kategoricke promenjive

levels(train$default.payment.next.month)<-make.names(levels(train$default.payment.next.month))
levels(test$default.payment.next.month)<-make.names(levels(test$default.payment.next.month))
trControl<-trainControl(method = "cv", number = 5,classProbs=TRUE)

# Kreiranje  modela, SVM sa linearnom Kernel funkcijom

model_svmlin <- train(default.payment.next.month ~ ., data = train, method = "svmLinear", trControl=trControl)

# Prikaz modela

model_svmlin # c=1, za lin metod.Ova konstanta predstavlja cost, misklasifikacije. Tako se sa vecom
# vrednosti konstante C, verovatnoca pogresne klasifikacije je manja.

# Najznacajniji prediktori za tacnost modela

impattr_svmlin<-data.frame(varImp(model_svmlin)[[1]])
impattr_svmlin<-rownames_to_column(impattr_svmlin)
impattr_svmlin[order(-abs(impattr_svmlin[,2])),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

# Predikcija

predict_svmlin<-predict(model_svmlin,test, type='prob')[,2]

# Brier score test

bs_svmlin<-round(BrierScore(as.numeric(test$default.payment.next.month)-1,predict_svmlin),3)

# AUC

roc_svmlin<-roc(test$default.payment.next.month,predict_svmlin)
svmlin_auc<-round(as.numeric(roc_svmlin$auc),3)
th<-coords(roc_svmlin, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_svmlin <- ifelse(predict_svmlin > th, 1,0)
cm_svmlin<-confusionMatrix(factor(predicted_svmlin),
                           factor(actuals),positive = "1")

# SVM Radial.  Kreiranje modela, SVM sa nelinearnom kernel funkcijom (Radial)

model_svmradial <- train(default.payment.next.month ~., data = train, method = "svmRadial",trControl=trControl)

# Hiperparametri  sigma and C, sa kojima se postize maksimalna tacnost modela

model_svmradial$bestTune

# Prikaz modela

plot(model_svmradial)

# Najznacajniji prediktori za tacnost modela

impattr_svmradial<-data.frame(varImp(model_svmradial)[[1]])
impattr_svmradial<-rownames_to_column(impattr_svmradial)
impattr_svmradial[order(-abs(impattr_svmradial[,2])),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

# Predikcija

predict_svmradial<-predict(model_svmradial, test,type='prob')[,2]

# Brier score test

bs_svmradial<-round(BrierScore(as.numeric(test$default.payment.next.month)-1,predict_svmradial),3)

# AUC

roc_svmradial<-roc(test$default.payment.next.month,predict_svmradial)
svmradial_auc<-round(as.numeric(roc_svmradial$auc),3)
th<-coords(roc_svmradial, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_svmradial <- ifelse(predict_svmradial > th, 1,0)
cm_svmradial<-confusionMatrix(factor(predicted_svmradial),
                              factor(actuals),positive = "1")

#  SVM Poli

# Kreiranje modela 

model_svmpoli <- train(default.payment.next.month ~ ., data = train, method = "svmPoly",trControl=trControl)

# Hiperparametri  sigma and C, sa kojima se postize maksimalna tacnost modela

model_svmpoli$bestTune

# Prikaz modela

plot(model_svmpoli)

# Najznacajniji prediktori za tacnost modela

impattr_svmpoli<-data.frame(varImp(model_svmpoli)[[1]])
impattr_svmpoli<-rownames_to_column(impattr_svmpoli)
impattr_svmpoli[order(-abs(impattr_svmpoli[,2])),][c(1:5),]# 5 najvecih reg koeficijenata po apsolutnoj vrednosti 

# Predikcija

predict_svmpoli<-predict(model_svmpoli,test, type = "prob")[,2]

# Brier score test

bs_svmpoli<-round(BrierScore(as.numeric(test$default.payment.next)-1,predict_svmpoli),3)

# AUC

roc_svmpoli<-roc(test$default.payment.next,predict_svmpoli)
svmpoli_auc<-round(as.numeric(roc_svmpoli$auc),3)
th<-coords(roc_svmpoli, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_svmpoli <- ifelse(predict_svmpoli > th, 1,0)
cm_svmpoli<-confusionMatrix(factor(predicted_svmpoli),
                            factor(actuals),positive = "1")

# SVM modeli rezultati

model_poredjenje<-data.frame(Model=c("SVM L","SVM R", "SVM P"),
                             Tacnost=c(round(cm_svmlin$overall[1],3),
                                       round(cm_svmradial$overall[1],3),
                                       round(cm_svmpoli$overall[1],3)),
                             
                             Sensitivity=c(round(cm_svmlin$byClass[1],3),
                                           round(cm_svmradial$byClass[1],3),
                                           round(cm_svmpoli$byClass[1],3)),
                             
                             Specificity=c(round(cm_svmlin$byClass[2],3),
                                           round(cm_svmradial$byClass[2],3),
                                           round(cm_svmpoli$byClass[2],3)),
                             
                             F1 =        c(round(cm_svmlin$byClass[7],3),
                                           round(cm_svmradial$byClass[7],3),
                                           round(cm_svmpoli$byClass[7],3)),
                             
                             Kappa=       c(round(cohen.kappa(cm_svmlin$table)$kappa,3),
                                            round(cohen.kappa(cm_svmradial$table)$kappa,3),
                                            round(cohen.kappa(cm_svmpoli$table)$kappa,3)),
                             
                             AUC=          c(svmlin_auc,svmradial_auc, svmpoli_auc),
                             
                             BS=          c(bs_svmlin,bs_svmradial,bs_svmpoli))

model_poredjenje%>%htmlTable(caption="Pokazatelji performansi SVM modela")

# AUC krive

roc(test$default.payment.next,predict_svmlin,plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.lab=1,cex.axis=1,cex.main=0.8, main="AUC, SVM modela",
    col="chocolate",identity.col="chocolate",print.thres.col="chocolate",print.auc.adj=c(0,0.5),print.thres.adj=-.10,print.thres.pch=10,print.auc.x=.5,print.auc.y=.49)
plot.roc(test$default.payment.next,predict_svmradial,print.auc = T, add = T,lwd= 1,lty=1,col="red",print.auc.x=.5,print.auc.y=.43)
plot.roc(test$default.payment.next,predict_svmpoli,print.auc = T, add = T,lwd= 1,lty=1,col="blue",print.auc.x=.5,print.auc.y=.34)
legend("topleft", legend=c("SVM L","SVM R","SVM P"), col=c("chocolate","red","blue"), lwd=1, cex=.8)


# 7. Heterogeni modeli

# Inicijalizacija h2o

h2o.init()

# Train i test h2o data frames
train_h2o<-default_cr_h2o[split,]
train_h2o<-ROSE(default.payment.next.month ~ ., data  = train_h2o)$data  
table(train_h2o$default.payment.next.month)# broj uzoraka ostaje isti


train_df_h2o<-as.h2o(train_h2o)
test_df_h2o<-as.h2o(default_cr_h2o[-split,])

# Odredjivanje zvisne i nezavisnih promenjivih 

y<-"default.payment.next.month"
x <- setdiff(names(train_df_h2o), y)


# 1. Konsolidacija 3 modela  (GBM (grad boosting) + Log + RF)

# GBM

prvi_gbm <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train_df_h2o,
                    nfolds = 5,
                    keep_cross_validation_predictions = TRUE,
                    seed = 5)

# Predikcija

predict_h2o_gbm<-as.vector(h2o.predict(prvi_gbm, test_df_h2o)[,3])
perf_gbm <- h2o.performance(prvi_gbm, newdata = test_df_h2o)

# ROC kriva

roc_h2o_gbm<-roc(actuals,predict_h2o_gbm)
th<-coords(roc_h2o_gbm, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_h2o_gbm <- ifelse(predict_h2o_gbm > th, 1,0)
cm_h2o_gbm<-confusionMatrix(factor(predicted_h2o_gbm),
                            factor(actuals),positive = "1")

auc_h2o_gbm <- h2o.auc(h2o.performance(prvi_gbm, newdata = test_df_h2o))

bs_h2o_gbm<-round(BrierScore(as.numeric(actuals)-1,predict_h2o_gbm),3)


drugi_glm <- h2o.glm(x = x,
                     y = y,
                     training_frame = train_df_h2o,
                     nfolds = 5,
                     keep_cross_validation_predictions = TRUE,
                     seed = 5)

# Predikcija

predict_h2o_glm<-as.vector(h2o.predict(drugi_glm, test_df_h2o)[,3])
perf_glm <- h2o.performance(drugi_glm, newdata = test_df_h2o)

# ROC kriva

roc_h2o_glm<-roc(actuals,predict_h2o_glm)
th<-coords(roc_h2o_glm, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_h2o_glm <- ifelse(predict_h2o_glm > th, 1,0)
cm_h2o_glm<-confusionMatrix(factor(predicted_h2o_glm),
                            factor(actuals),positive = "1")

auc_h2o_glm <- h2o.auc(h2o.performance(drugi_glm, newdata = test_df_h2o))
bs_h2o_glm<-round(BrierScore(as.numeric(actuals)-1,predict_h2o_glm),3)


# RF

treci_rf <- h2o.randomForest(x = x,
                             y = y,
                             training_frame = train_df_h2o,
                             nfolds = 5,
                             keep_cross_validation_predictions = TRUE,
                             seed = 5)


# Predikcija

predict_h2o_rf<-as.vector(h2o.predict(treci_rf, test_df_h2o)[,3])
perf_rf <- h2o.performance(treci_rf, newdata = test_df_h2o)

# ROC kriva

roc_h2o_rf<-roc(actuals,predict_h2o_rf)
th<-coords(roc_h2o_rf, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_h2o_rf <- ifelse(predict_h2o_rf > th, 1,0)
cm_h2o_rf<-confusionMatrix(factor(predicted_h2o_rf),
                           factor(actuals),positive = "1")

auc_h2o_rf <- h2o.auc(h2o.performance(treci_rf, newdata = test_df_h2o))
bs_h2o_rf<-round(BrierScore(as.numeric(actuals)-1,predict_h2o_rf),3)


# Objedinjavanje pojedinacnih predikcija, stacking

ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                metalearner_algorithm="drf",
                                training_frame = train_df_h2o,
                                base_models = list(prvi_gbm, drugi_glm,treci_rf))

#Predikcija

predict_ensemb<-as.vector(h2o.predict(ensemble, test_df_h2o)[,3])
roc_ensemb<-roc(actuals,predict_ensemb)
th<-coords(roc_ensemb, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_ensemb <- ifelse(predict_ensemb > th, 1,0)
cm_h2o_ensemb<-confusionMatrix(factor(predicted_ensemb),
                               factor(actuals),positive = "1")


# Poredjenje performansi 

perf_rf_test <- h2o.performance(treci_rf, newdata = test_df_h2o)
perf_glm_test <- h2o.performance(drugi_glm, newdata = test_df_h2o)
perf_gbm_test<-  h2o.performance(prvi_gbm, newdata = test_df_h2o)
perf_ensem_test <- h2o.performance(ensemble, newdata = test_df_h2o)


model_poredjenje<-data.frame(Model=c("RF_h2o","GLM_h2o","GBM_h2o","Ensemb"),
                             
                             Tacnost=c(round(cm_h2o_rf$overall[1],3),
                                       round(cm_h2o_glm$overall[1],3),
                                       round(cm_h2o_gbm$overall[1],3),
                                       round(cm_h2o_ensemb$overall[1],3)),
                             
                             Sensitivity=c(round(cm_h2o_rf$byClass[1],3),
                                           round(cm_h2o_glm$byClass[1],3),
                                           round(cm_h2o_gbm$byClass[1],3),
                                           round(cm_h2o_ensemb$byClass[1],3)),
                             
                             Specificity=c(round(cm_h2o_rf$byClass[2],3),
                                           round(cm_h2o_glm$byClass[2],3),
                                           round(cm_h2o_gbm$byClass[2],3),
                                           round(cm_h2o_ensemb$byClass[2],3)),
                             
                             F1 =        c(round(cm_h2o_rf$byClass[7],3),
                                           round(cm_h2o_glm$byClass[7],3),
                                           round(cm_h2o_gbm$byClass[7],3),
                                           round(cm_h2o_ensemb$byClass[2],3)),
                             
                             Kappa=       c(round(cohen.kappa(cm_h2o_rf$table)$kappa,3),
                                            round(cohen.kappa(cm_h2o_glm$table)$kappa,3),
                                            round(cohen.kappa(cm_h2o_gbm$table)$kappa,3),
                                            round(cohen.kappa(cm_h2o_ensemb$table)$kappa,3)),
                             
                             AUC= c(round(h2o.auc(perf_rf_test),3),round(h2o.auc(perf_glm_test),3),round(h2o.auc(perf_gbm_test),3), round(h2o.auc(perf_ensem_test),3)))

model_poredjenje%>%htmlTable(caption="Pokazatelji performansi modela, primenom H2o")

# Modeli sa najvecim AUC

roc(test$default.payment.next.month,predict_h2o_gbm,plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.main=0.8, main="Modeli sa najvecom, AUC",
    col="chocolate",identity.col="chocolate",print.thres.col="chocolate",print.auc.x=.5,print.auc.y=.49)
plot.roc(test$default.payment.next.month,predict_h2o_rf,print.auc = T, add = T,lwd= 1,lty=1,col="red",print.auc.x=.5,print.auc.y=.41)
plot.roc(test_norm$default.payment.next.month,predict_ridge,print.auc = T, add = T,lwd= 1,lty=1,col="blue",print.auc.x=.5,print.auc.y=.33)
legend("topleft", legend=c("GTB","RF","LR Ridge"), col=c("chocolate","red","blue"), lwd=1, cex=.8)


# Kraj





