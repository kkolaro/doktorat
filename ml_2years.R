
# ANALIZA PERFORMANSI KLASIFIKACIONIH ALGORITAMA

library(imbalance)
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
library(klaR)
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
library(kernlab)


# 1. FORMIRANJE  DATA SETA

# Import podataka

stecaj<-read.arff(choose.files())# file 4, stecaj posle 2 godine

# Promena imena kolone zavisne binarne promenjive 'class', u 'Stecaj'

colnames(stecaj)[65]<-'Stecaj'

# Provera balansiranosti podataka

round (prop.table( as.table(table(stecaj$Stecaj))),3)

# 2. OPIS PODATAKA

# Descriptivna statistika data seta.
# Mean, Median, Q1, Q3, Min, Max, prisutvo NA's

summary(stecaj)

# 3. ZAMENA NEPOSTOJECIH PODATAKA NAs 

broj_NAs<-sum(is.na(stecaj))

if (broj_NAs != 0) {
  
  cat("Ukupan broj NAs =",broj_NAs,"\n")
  
} else {
  
  print("Data set je kompletan")
}

# Graficka provera postojanja NAs i sablona pojavljivanja,primenom  mice package 

md.pattern(stecaj, plot = F) # Sablon pojavljivanja NAs. mice package. 
missmap(stecaj)# Amelia package 

# Provera da li zavisna promenjiva 'Stecaj' ima NA's

if (sum(is.na(stecaj[,65]))==0) { 
  
  cat("Zavisna promenjiva 'Stecaj' nema NA","\n")
}

# Unos zamenskih vrednosti za NAs primenom RF 

stecaj_bezNA<-rfImpute(Stecaj~.,data=stecaj, iter=2)
sum(is.na(stecaj_bezNA))
missmap(stecaj_bezNA)

# 4. PREGLED DISTRIBUCIJA PREDIKTORA 

opis<-describe(stecaj_bezNA) # psych
head(opis)

# Visoke negativne i pozitivne vrednosti skewness  ukazuju na nesimetricnu distribuciju podataka.
# Visoke vrednosti kurtosis oznacavaju Leptokurtske distribucije za koje je karaktereisticno 
# da se vrhovi distribucije tanji i visoki,a repovi zadebljani.

normal_prediktor<-subset(opis,(skew<=1&skew>=-1)&(kurtosis<3&kurtosis>-3),select=c(skew,kurtosis))
normal_prediktor
cat("Prediktori sa priblizno normalnom distribucijom:  ",row.names(normal_prediktor)[],"\n")

hist(stecaj_bezNA$Attr29)

# 5. PROVERA POSTOJANJA MULTIKOLINEARNOSTI 

corelacija<-cor(stecaj_bezNA[,2:64])# korelacija prediktora
corrplot(corelacija,method='number',type="upper",order="hclust",tl.col='black',tl.pos='n') # plava boja, pozitivna korelacija, crvena negativna. Intezitet boje odgovara intezitetu lin zavisnosti
corr_cross(stecaj_bezNA[,2:64], max_pvalue = 0.05, top = 10)# package lares. Parovi 10 znacajnih korelacija(p-value<0.05) U plavoj boji su pozitivne korelacije. Korelacije vece od 5%

# Sledi da postoji korelacija (pozitivna i negativna),izmedju prediktora

# 6. ANALIZA POSTOJANJA EKSTREMNIH VREDNOSTI  

# Vrednosti prediktora koje se nalaze van granice min i max whisker-a, ce se smatrati outliers i bice zamenjenjene
# sa median vrednosti odgovarajuceg prediktora

stecaj_bezekstrema_bezNA<-stecaj_bezNA

for(i in 2:65){
  
  out<-boxplot.stats(stecaj_bezNA[,i],coef = 1.5)$out
  out_ind <- which(stecaj_bezNA[,i] %in% c(out))
  stecaj_bezekstrema_bezNA[c(out_ind),i]<-median(stecaj_bezNA[,i], na.rm = TRUE)
}

summary(stecaj_bezekstrema_bezNA)

# Odredjivanje znacaja prediktora van ML modela, primenom biserialne korelacije (Pearson, 1909)

znacaj_prediktora<-sapply(2:65, function(i) biserial.cor(stecaj_bezekstrema_bezNA[,i],stecaj_bezekstrema_bezNA$Stecaj))
atributi<-c(1:64)

vaznost_attr<-data.frame(Atributi=atributi,Znacaj_prediktora=round(znacaj_prediktora,3))
vaznost_attr<-vaznost_attr[order(abs(vaznost_attr$Znacaj_prediktora), decreasing = TRUE), ][1:5,]
rownames(vaznost_attr) <- NULL
vaznost_attr%>%htmlTable(caption="Uticaj prediktora (aps. vrednost) na vrednost zavisne promenjive, Biserialna korelacija")   

# Scaling podataka

preproc <- preProcess(stecaj_bezekstrema_bezNA[,-1], method=c("center","scale"))
stecaj_bezekstrema_bezNA_scaled <- predict(preproc, stecaj_bezekstrema_bezNA[,-1])
stecaj_bezekstrema_bezNA_scaled$Stecaj<- as.factor(stecaj_bezekstrema_bezNA$Stecaj)
summary(stecaj_bezekstrema_bezNA_scaled)

# 6.  KREIRANJE TRENING/TESTING PODATAKA

# Train/test podaci u odnosu 70/30

split <- createDataPartition (stecaj_bezekstrema_bezNA_scaled$Stecaj, p = .7, list = F)# caret

train <- stecaj_bezekstrema_bezNA_scaled[split,]# Attr1....Attr64, Stecaj
test  <- stecaj_bezekstrema_bezNA_scaled[-split,]

train<-train[sample(1:nrow(train)),]
test<-test[sample(1:nrow(test)),]

actuals<-test$Stecaj# sacuvacemo originalne vrednosti za K NN i SVM, potrebne usled naknadnih transformacija u ovim algoritmima


# 7. KREIRANJE PRINCIPALNIH KOMPONENTI pc.


pc<-prcomp(select_if(stecaj_bezekstrema_bezNA_scaled,is.numeric)) # kreiranje PC1...PC64, u funkciji Attr1....Attr64

res.pca<-PCA(select_if(stecaj_bezekstrema_bezNA_scaled,is.numeric),grap=F)
fviz_eig(res.pca,addlabels = T,ncp=10)# varijacija explained sa svakom PC

corelacija_pc<-cor(pc$x)# PC su nezavisne
corrplot(corelacija_pc,type="full",order="hclust",tl.col='black',tl.pos='n') 

cat("Broj PC sa vrednostima eigenvalue >= jedan je",max(which (get_eig(pc)[,1]>=1,)),"\n")
pc_95<-which(get_eig(pc)[[3]]>95)[1]#PC sa kojom se moze objasniti 95%  cumulatvne  var
stecaj_pc<-cbind(data.frame(pc$x)[,1:pc_95],Stecaj=stecaj_bezekstrema_bezNA_scaled$Stecaj)

#Train/test podaci u funkciji PCs

train_pc <- stecaj_pc[split,]
test_pc  <- stecaj_pc[-split,]

train_pc<-train_pc[sample(1:nrow(train_pc)),]
test_pc<-test_pc[sample(1:nrow(test_pc)),]


# 8. PREDIKTIVNI MODELI

# 8.1 LOGISTICKA REGRESIJA

# Kreiranje modela

model_lr<-glm(Stecaj~ ., data=train, family = binomial)
summary(model_lr)

# Najvazniji, najinformativniji prediktori

vip(model_lr,num_features = 5, method= "model")

# Predikcija

predict_lr<-model_lr%>%predict(test,type="response")

# Brier Score test, od 0 do 1. Manja vrednost, bolji model

bs_lr<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_lr),3)

# AUC

roc_lr<-roc(test$Stecaj,predict_lr)
lr_auc<-round(as.numeric(roc_lr$auc),3)
th<-coords(roc_lr, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_lr<-ifelse(predict_lr>th,"1","0")
cm_lr<-confusionMatrix(factor(predicted_lr), test$Stecaj, positive = '1',mode = "everything")

# 8.1.1 Log regresija sa PC

model_lrpc<-glm(Stecaj~ ., data=train_pc, family = binomial)#model sa PC
summary(model_lrpc)

# Predikcija

predict_lrpc<-model_lrpc%>%predict(test_pc,type="response")# sa PC

# Brier Score test

bs_lrpc<-round(BrierScore(as.numeric(test_pc$Stecaj)-1, predict_lrpc),3)

# AUC

roc_lr_pc<-roc(test_pc$Stecaj,predict_lrpc)
lr_pc_auc<-round(as.numeric(roc_lr_pc$auc),3)
th<-coords(roc_lr_pc, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_lrpc<-ifelse(predict_lrpc>th,"1","0")
cm_lrpc<-confusionMatrix(factor(predicted_lrpc), test_pc$Stecaj, positive = '1',mode = "everything")

# 8.1.1 PENALIZED LOGISTIC REGRESSION, regularizacija modela (ne proredimo prema AIC)

#Lasso  regresija

# Kreiranje matrice prediktora 

x_train<- model.matrix(Stecaj~., data=train,)[,-1]
x_test<-model.matrix(Stecaj~., test,)[,-1]

y_output<-train$Stecaj

# Koriscenjem CV trazimo najoptimalniju vrednost lambda, kojom se odredjuje stepen 'sankcionisanja' kompleksnosti modela tj. umanjenje reg koeficijenata 

lasso_param<-cv.glmnet(x_train,y_output,alpha=1,family="binomial", nfolds = 5)# glmnet package.  Za alpha 0=1, radi se o Lasso reg

# Kreiranje modela

model_lrlasso<- glmnet(x_train,y_output, alpha = 1, family = "binomial",lambda = lasso_param$lambda.min)

# Predikcija

predict_lrlasso<-model_lrlasso%>%predict(newx=x_test,type="response")

# Brier Score test

bs_lasso<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_lrlasso),3)

# AUC

roc_lasso<-roc(test$Stecaj,as.vector(predict_lrlasso))
lasso_auc<-round(as.numeric(roc_lasso$auc),3)
th<-coords(roc_lasso, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_lrlasso<-ifelse(predict_lrlasso> th,"1","0")
cm_lasso<-confusionMatrix(factor(predicted_lrlasso), test$Stecaj, positive = '1',mode = "everything")

# Najvazniji, najinformativniji prediktori

vip(model_lrlasso,num_features = 5, method= "model")

# Ridge regresija

ridge_param<-cv.glmnet(x_train,y_output,alpha=0,family="binomial",nfolds = 5)# glmnet package.  Za alpha =0 , radi se o Ridge reg

# Kreiranje modela

model_lrridge<- glmnet(x_train,y_output, alpha = 0, family = "binomial",lambda = ridge_param$lambda.min)

# Predikcija

predict_lrridge<-model_lrridge%>%predict(newx=x_test,type="response")

# Brier score test

bs_ridge<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_lrridge),3)

# AUC

roc_ridge<-roc(test$Stecaj,as.vector(predict_lrridge))
ridge_auc<-round(as.numeric(roc_ridge$auc),3)
th<-coords(roc_ridge, "best", ret="threshold")$threshold  #pROC

#Matrica konfuzije

predicted_lrridge<-ifelse(predict_lrridge>th,"1","0")
cm_ridge<-confusionMatrix(factor(predicted_lrridge), test$Stecaj, positive = '1',mode = "everything")

# Najvazniji, najinformativniji prediktori

vip(model_lrridge,num_features = 5, method= "model")

# Elastic net regresija

elasticnet_param<-cv.glmnet(x_train,y_output,alpha=0.7,family="binomial",nfolds = 5)# glmnet package. Za alpha izmedju 0 i 1, radi se o Elastic Net regresiji

# Kreiranje modela

model_elasnet<- glmnet(x_train,y_output, alpha = 0.7, family = "binomial",lambda = elasticnet_param$lambda.min)

# Predikcija

predict_elasnet<-model_elasnet%>%predict(newx=x_test,type="response")

# Brier score test

bs_elasnet<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_elasnet),3)

# AUC

roc_elasnet<-roc(test$Stecaj,as.vector(predict_elasnet))
elasnet_auc<-round(as.numeric(roc_elasnet$auc),3)
th<-coords(roc_elasnet, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_elasnet<-ifelse(predict_elasnet>th,"1","0")
cm_elsnet<-confusionMatrix(factor(predicted_elasnet), test$Stecaj, positive = '1',mode = "everything")

# Najvazniji, najinformativniji prediktori

vip(model_elasnet,num_features = 5, method= "model")

# Poredjenje linearni regresioni modelo

model_poredjenje<-data.frame(Model=c("LinReg","LinReg_PC","Lasso","Ridge","ElasNet"),
                             
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
                             
                             AUC   =c(lr_auc,lr_pc_auc,lasso_auc,ridge_auc,elasnet_auc),
                             
                             BS=c(bs_lr,bs_lrpc, bs_lasso,bs_ridge,bs_elasnet))
                            

model_poredjenje%>%htmlTable(caption="Pokazatelji performansi regresionih modela")   

# ROC krive

roc(test$Stecaj,as.vector(predict_lr),plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.main=0.8, main="AUC, modela Linearne regresije",
            col="yellow",identity.col="yellow",print.thres.col="yellow",print.thres.pch=10,print.auc.x=.5,print.auc.y=.45)

plot.roc(test_pc$Stecaj,as.vector(predict_lrpc),print.auc = T, add = T,lwd= 1,lty=1,col="brown",print.auc.x=.5,print.auc.y=.38)
plot.roc(test$Stecaj,as.vector(predict_lrlasso),print.auc = T, add = T,lwd= 1,lty=1,col="red",print.auc.x=.5,print.auc.y=.31)
plot.roc(test$Stecaj,as.vector(predict_lrridge),print.auc = T, add = T,lwd= 1,lty=1,col="blue",print.auc.x=.5,print.auc.y=.24)
plot.roc(test$Stecaj,as.vector(predict_elasnet),print.auc = T, add = T,lwd= 1,lty=1,col="green",print.auc.x=.5,print.auc.y=.17)

legend("topleft", legend=c("LR","LRpc","Lasso","Ridge","ElasNet"), col=c("yellow","brown","red","blue","green"), lwd=1, cex=.8)

# Statisticki znacaj AUC razlika

roc.test(roc_ridge,roc_lr,method=c("delong"))# najveca razlika izmedju dve AUC povrsine nije statisticki znacajna

# 8.1.2 LDA i QDA (koriscenjem prediktora najtacnijeg lasso regresionog modela)


# LDA

# Kreiranje modela

model_lda<-lda(Stecaj~ ., data=train)

par(mar=c(1,1,1,1))
plot(model_lda)# x osa je LD1, prvi grafik za klasu 0 , drugi klasu 1

# Predikcija

predict_lda<-model_lda%>%predict(test)
names(predict_lda)# class uzorka, posterior pr da uyorak pripda nekoj od klasa, x su LD
head(predict_lda$posterior,2)

# Brier score test

bs_lda<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_lda$posterior[,2]),3)

# AUC

roc_lda<-roc(test$Stecaj,predict_lda$posterior[,2])
lda_auc<-round(as.numeric(roc_lda$auc),3)
th<-coords(roc_lda, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_lda<-ifelse(predict_lda$posterior[,2] > th,"1","0")
cm_lda<-confusionMatrix(factor(predicted_lda), test$Stecaj, positive = '1')

# Najvazniji, najinformativniji prediktori

coef(model_lda)# Intercept i 64 prediktora
df_regkoeficijenti<-data.frame(coef(model_lda))
impattr_lda<-rownames_to_column(df_regkoeficijenti)# package tibble
colnames(impattr_lda)<-c('Prediktori','RegKoeficijenti')
impattr_lda[order(-abs(impattr_lda$RegKoeficijenti)),][c(1:5),]#5 najznacajnijih  prediktora po apsolutnoim vrednostima

# QDA

model_qda<-qda(Stecaj~ .   -Attr18 , data=train) # model javlja gresku sa Attr 18

# Predikcija

predict_qda<-model_qda%>%predict(test)

# Brier score test

bs_qda<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_qda$posterior[,2]),3)

# AUC

roc_qda<-roc(test$Stecaj,predict_qda$posterior[,2])
qda_auc<-round(as.numeric(roc_qda$auc),3)
th<-coords(roc_qda, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_qda<-ifelse(predict_qda$posterior[,2] >th,"1","0")
cm_qda<-confusionMatrix(factor(predicted_qda), test$Stecaj, positive = '1')

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

# ROC krive

roc(test$Stecaj,predict_lda$posterior[,2],plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.main=0.8, main="AUC, LDA i QDA ",
    col="yellow",identity.col="yellow",print.thres.col="yellow",print.auc.x=.5,print.auc.y=.43)
plot.roc(test$Stecaj,predict_qda$posterior[,2],print.auc = T, add = T,lwd= 1,lty=1,col="red",print.auc.x=.5,print.auc.y=.36)
legend("topleft", legend=c("LDA","QDA"), col=c("yellow","red"), lwd=1, cex=.8)

# Statisticki znacaj AUC razlika

roc.test(roc_lda,roc_qda,method=c("delong"))# nije statisticki znacajna/


# 8.2 STABLA ODLUCIVANJA

trControl=trainControl("cv",number=5 ) # sampling = "smote",Balansiranje trening podataka ML modela stabla sa  sampaling opcijama  smote 

# Kreiranje modela

model_tree<-train(Stecaj~., data=train,method="rpart",trControl= trControl, tuneLength=10)# tuneLength daje moguce cp vrednosti
ggplot(model_tree)# mala vrednost complexity parametra , dublje stablo
model_tree$bestTune

# Najinformativniji prediktori

vip(model_tree, num_features = 5, bar=F)

# Predikcija primenom DT

predict_tree<-data.frame(predict(model_tree,test,type="prob"))

# Brier score test

bs_tree<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_tree[,2]),3)

# AUC

roc_tree<-roc(test$Stecaj,predict_tree[,2])
tree_auc<-round(as.numeric(roc_tree$auc),3)
th<-coords(roc_tree, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_tree <- ifelse(predict_tree[,2] > th, 1,0)
cm_tree<-confusionMatrix(factor(predicted_tree),
                         test$Stecaj,positive = "1")


# 8.3 RANDOM FOREST

# Kreiranje modela

model_rf<-train(Stecaj~., data=train,method="rf",trControl= trControl,tuneLength=10)
plot(model_rf)
model_rf$bestTune # mtry br. prediktora za svako stablo
cat("Optimalan broj prediktora koji ce se koristiti pri izradi stabala je" ,model_rf$finalModel$mtry )

model_rf$finalModel 

# Najinformativniji prediktori

vip(model_rf, num_features = 5, bar=F)

# Predikcija primenom RF

predict_rf<-data.frame(predict(model_rf,test,type="prob"))

# Brier score test

bs_rf<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_rf$X1),3)

# AUC

roc_rf<-roc(test$Stecaj,predict_rf$X1)
rf_auc<-round(as.numeric(roc_rf$auc),3)
th<-coords(roc_rf, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_rf <- ifelse(predict_rf$X1 > th, 1,0)
cm_rf<-confusionMatrix(factor(predicted_rf),
                         test$Stecaj,positive = "1")

# 8.4 BOOSTING MODEL, extreme gradient boosting. Koristi sve corove procesora, paralelno procesiranje

model_xgb<-train(Stecaj~., data=train,method="xgbTree",trControl= trControl,verbosity = 0)
model_xgb$bestTune # mtry br. prediktora za svako stablo

# Najinformativniji prediktori

vip(model_xgb, num_features = 5, bar=F)

# Predikcija primenom DT

predict_xgb<-data.frame(predict(model_xgb,test,type="prob"))

# Brier score test

bs_xgb<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_xgb$X1),3)

# AUC

roc_xgb<-roc(test$Stecaj,predict_xgb$X1)
xgb_auc<-round(as.numeric(roc_xgb$auc),3)
th<-coords(roc_xgb, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_xgb <- ifelse(predict_xgb$X1 > th, 1,0)
cm_xgb<-confusionMatrix(factor(predicted_xgb),
                        test$Stecaj,positive = "1")



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

model_poredjenje%>%htmlTable(caption="Pokazatelji performansi modela na bazi stabla, balansirani podaci sa SMOTE")

# ROC krive

roc(test$Stecaj,predict_tree[,2],plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.main=0.8, main="AUC, modela na bazi stabla",
    col="yellow",identity.col="yellow",print.thres.col="yellow",print.auc.x=.5,print.auc.y=.49)
plot.roc(test$Stecaj,predict_rf$X1,print.auc = T, add = T,lwd= 1,lty=1,col="red",print.auc.x=.5,print.auc.y=.41)
plot.roc(test$Stecaj,predict_xgb$X1,print.auc = T, add = T,lwd= 1,lty=1,col="blue",print.auc.x=.5,print.auc.y=.33)

legend("topleft", legend=c("DT","RF","GTB"), col=c("yellow","red","blue"), lwd=1, cex=.8)

# Statisticki znacaj AUC razlika

roc.test(roc_tree,roc_xgb,method=c("delong"))# statisticki znacajna razlika


# 8.5 NB

# Kreiranje modela

model_nb<-train(Stecaj~., data=train,method="nb",trControl=trControl)
model_nb # vidimo mean i sd za svaki numericki prediktor razlicitih klasa

# Predikcija

predict_nb<-predict(model_nb,test,type='prob')[,2]

# Brier score test

bs_nb<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_nb),3)

# AUC

roc_nb<-roc(test$Stecaj,predict_nb)
nb_auc<-round(as.numeric(roc_nb$auc),3)
th<-coords(roc_nb, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_nb <- ifelse(predict_nb > th, 1,0)
cm_nb<-confusionMatrix(factor(predicted_nb),
                            test$Stecaj,positive = "1")

# Performanse NB

model_poredjenje<-data.frame(Model="NB",Tacnost= round(cm_nb$overall[1],3),
                             
                             Sensitivity=round(cm_nb$byClass[1],3),
                             Specificity=round(cm_nb$byClass[2],3),
                             F1 =round(cm_nb$byClass[7],3),
                             Kappa= round(cohen.kappa(cm_nb$table)$kappa,3),
                             AUC   = round(as.numeric(roc_nb$auc),3),
                             BS= bs_nb)    

model_poredjenje%>%htmlTable(caption="Pokazatelji performansi NB modela")

# ROC kriva

roc(test$Stecaj,predict_nb,plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.main=0.8, main="AUC, NB",
    col="red",identity.col="red",print.thres.col="red",print.auc.x=.5,print.auc.y=.49)

# 8.6 KNN model

# Kreiranje sintaticka ispravnih imena

levels(train$Stecaj)<-make.names(levels(train$Stecaj))
levels(test$Stecaj)<-make.names(levels(test$Stecaj))

trControl<-trainControl(method = "repeatedcv", number = 5,repeats=3,classProbs = T,summaryFunction = twoClassSummary)

# Kreiranje modela 

model_knn<-train(Stecaj~., data = train, method='knn',tuneLength=10, 
                  trControl=trControl,metric="ROC")

plot(model_knn,xlab="Broj suseda K",ylab='Tacnost u funkciji K', type='b',col='black',lwd=1.5,pch='o')

model_knn$bestTune

# Vaznost prediktora

impattr_knn<-data.frame(varImp(model_knn)[[1]])
impattr_knn<-rownames_to_column(impattr_knn)# package tibble
colnames(impattr_knn)<-c('Prediktori','RegKoeficijenti')
impattr_knn[order(-abs(impattr_knn$RegKoeficijenti)),][c(1:5),]# 5 najznacajnijih  prediktora po apsolutnoim vrednostima

# Predikcija

predict_knn<-predict(model_knn,test,type='prob')[,2]

# Brier score test

bs_knn<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_knn),3)

# AUC

roc_knn<-roc(test$Stecaj,predict_knn)
knn_auc<-round(as.numeric(roc_knn$auc),3)
th<-coords(roc_knn, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_knn <- ifelse(predict_knn > th, 1,0)
cm_knn<-confusionMatrix(factor(predicted_knn),
                        factor(actuals) ,positive = "1")

# ROC kriva

roc(test$Stecaj,predict_knn,plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.main=0.8, main="AUC, KNN",
    col="red",identity.col="red",print.thres.col="red",print.auc.x=.5,print.auc.y=.49)


# Performanse K NN

model_poredjenje<-data.frame(Model=c("K-NN"),
                             Tacnost=c(round(cm_knn$overall[1],3)),
                             
                             Sensitivity=c(round(cm_knn$byClass[1],3)),
                             
                             Specificity=c(round(cm_knn$byClass[2],3)),
                             
                             F1 =        c(round(cm_knn$byClass[7],3)),
                             
                             Kappa=       c(round(cohen.kappa(cm_knn$table)$kappa,3)),
                             
                             
                             AUC=          c(knn_auc),
                             
                             BS=          c(bs_knn))

model_poredjenje%>%htmlTable(caption="Pokazatelji performansi K-NN modela")



# 8.7  SUPORT VECTOR MACHINE 

# 8.7.1 SVM linear


trControl<-trainControl(method = "cv", number = 5,classProbs=TRUE)

# Kreiranje  modela, SVM sa linearnom Kernel funkcijom

model_svmlin <- train(Stecaj ~ ., data = train, method = "svmLinear", trControl=trControl)

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

bs_svmlin<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_svmlin),3)

# AUC

roc_svmlin<-roc(test$Stecaj,predict_svmlin)
svmlin_auc<-round(as.numeric(roc_svmlin$auc),3)
th<-coords(roc_svmlin, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_svmlin <- ifelse(predict_svmlin > th, 1,0)
cm_svmlin<-confusionMatrix(factor(predicted_svmlin),
                           factor(actuals),positive = "1")


# 8.7.2  SVM Radial

model_svmradial <- train(Stecaj ~., data = train, method = "svmRadial", trControl=trControl, tuneLength = 5,metric="ROC")

# Hiperparametri  sigma and C, sa kojima se postize maksimalna tacnost modela

model_svmradial$bestTune

# Prikaz modela

plot(model_svmradial)

# Najznacajniji prediktori za tacnost modela

impattr_svmradial<-data.frame(varImp(model_svmradial)[[1]])
impattr_svmradial<-rownames_to_column(impattr_svmradial)
impattr_svmradial[order(-abs(impattr_svmradial[,2])),][c(1:5),]# 5 najznacajnijih  prediktora po apsolutnoim vrednostima 

# Predikcija

predict_svmradial<-predict(model_svmradial,test,type='prob')[,2]

# Brier score test

bs_svmradial<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_svmradial),3)

# AUC

roc_svmradial<-roc(test$Stecaj,predict_svmradial)
svmradial_auc<-round(as.numeric(roc_svmradial$auc),3)
th<-coords(roc_svmradial, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_svmradial <- ifelse(predict_svmradial > th, 1,0)
cm_svmradial<-confusionMatrix(factor(predicted_svmradial),
                           factor(actuals),positive = "1")

# 8.7.3 SVM Poli

# Kreiranje modela, SVM sa nelinearnom kernel funkcijom (polinomalna)

model_svmpoli <- train(Stecaj ~., data = train, method = "svmPoly", metric="ROC",trControl=trControl)

# Hiperparametri  sigma and C, sa kojima se postize maksimalna tacnost modela

model_svmpoli$bestTune

# Prikaz modela

plot(model_svmpoli)

# Najznacajniji prediktori za tacnost modela

impattr_svmpoli<-data.frame(varImp(model_svmpoli)[[1]])
impattr_svmpoli<-rownames_to_column(impattr_svmpoli)
impattr_svmpoli[order(-abs(impattr_svmpoli[,2])),][c(1:5),]# 5 najznacajnijih  prediktora po apsolutnoim vrednostima 

# Predikcija

predict_svmpoli<-predict(model_svmpoli,test,type='prob')[,2]

# Brier score test

bs_svmpoli<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_svmpoli),3)

# AUC

roc_svmpoli<-roc(test$Stecaj,predict_svmpoli)
svmpoli_auc<-round(as.numeric(roc_svmpoli$auc),3)
th<-coords(roc_svmpoli, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_svmpoli <- ifelse(predict_svmpoli > th, 1,0)
cm_svmpoli<-confusionMatrix(factor(predicted_svmpoli),
                           factor(actuals),positive = "1")


# Performanse SVM modela

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

# ROC krive

roc(test$Stecaj,predict_svmlin,plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.lab=1,cex.axis=1,cex.main=0.8, main="AUC, SVM modeli",
    col="yellow",identity.col="yellow",print.thres.col="yellow",print.auc.adj=c(0,0.5),print.thres.adj=-.10,print.thres.pch=10,print.auc.x=.5,print.auc.y=.49)
plot.roc(test$Stecaj,predict_svmradial,print.auc = T, add = T,lwd= 1,lty=1,col="red",print.auc.x=.5,print.auc.y=.44)
plot.roc(test$Stecaj,predict_svmpoli,print.auc = T, add = T,lwd= 1,lty=1,col="blue",print.auc.x=.5,print.auc.y=.37)
legend("topleft", legend=c("SVM L","SVM R","SVM P"), col=c("yellow","red","blue"), lwd=1, cex=.8)


# 9 Heterogeni modeli

# Inicijalizacija h2o

h2o.init()

# Train i test h2o data frames

train_df_h2o<-as.h2o(train)
test_df_h2o<-as.h2o(test)

# Odredjivanje zavisne i nezavisnih promenjivih 

y<-"Stecaj"
x <- setdiff(names(train), y)


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

roc_h2o_gbm<-roc(test$Stecaj,predict_h2o_gbm)
th<-coords(roc_h2o_gbm, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_h2o_gbm <- ifelse(predict_h2o_gbm > th, 1,0)
cm_h2o_gbm<-confusionMatrix(factor(predicted_h2o_gbm),
                           factor(actuals),positive = "1")

auc_h2o_gbm <- h2o.auc(h2o.performance(prvi_gbm, newdata = test_df_h2o))

bs_h2o_gbm<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_h2o_gbm),3)


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

roc_h2o_glm<-roc(test$Stecaj,predict_h2o_glm)
th<-coords(roc_h2o_glm, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_h2o_glm <- ifelse(predict_h2o_glm > th, 1,0)
cm_h2o_glm<-confusionMatrix(factor(predicted_h2o_glm),
                            factor(actuals),positive = "1")

auc_h2o_glm <- h2o.auc(h2o.performance(drugi_glm, newdata = test_df_h2o))
bs_h2o_glm<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_h2o_glm),3)


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

roc_h2o_rf<-roc(test$Stecaj,predict_h2o_rf)
th<-coords(roc_h2o_rf, "best", ret="threshold")$threshold  #pROC

# Matrica konfuzije

predicted_h2o_rf <- ifelse(predict_h2o_rf > th, 1,0)
cm_h2o_rf<-confusionMatrix(factor(predicted_h2o_rf),
                            factor(actuals),positive = "1")

auc_h2o_rf <- h2o.auc(h2o.performance(treci_rf, newdata = test_df_h2o))
bs_h2o_rf<-round(BrierScore(as.numeric(test$Stecaj)-1,predict_h2o_rf),3)


# Objedinjavanje pojedinacnih predikcija, stacking

ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                metalearner_algorithm="drf",
                                training_frame = train_df_h2o,
                                base_models = list(prvi_gbm, drugi_glm,treci_rf))

#Predikcija

predict_ensemb<-as.vector(h2o.predict(ensemble, test_df_h2o)[,3])
roc_ensemb<-roc(test$Stecaj,predict_ensemb)
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

model_poredjenje%>%htmlTable(caption="Pokazatelji performansi primenom h2o paketa: GLM, RF,GBM i heterogeni ensembling model")


# Tri najbolji modeli prema AUC



roc(test$Stecaj,as.vector(predict_lr),plot=T,print.auc = T,asp = NA,lwd = 1,auc.polygon=T,lty=1,cex.lab=1,cex.axis=1,cex.main=0.8, main="Najbolji modeli prema AUC",
    col="green",identity.col="green",print.thres.col="green",print.auc.adj=c(0,0.5),print.thres.adj=-.10,print.thres.pch=10,print.auc.x=.5,print.auc.y=.49)
plot.roc(test$Stecaj,predict_xgb$X1, print.auc = T, add = T,lwd= 1,lty=1,col="blue",print.auc.x=.5,print.auc.y=.445)
plot.roc(test$Stecaj,predict_rf$X1,print.auc = T, add = T,lwd= 1,lty=1,col="brown",print.auc.x=.5,print.auc.y=.37)


legend("topleft", legend=c("LogR","xGB","Rf"), col=c("green","blue","brown"), lwd=1, cex=.8)


# 10. Z SCORE  predikcija 

# Podaci za z score, tacnost predikcije,  nisu scaled

test_noscale<-stecaj_bezekstrema_bezNA[-split,c(1,4,7,8,9,10)]


# a je Attr3 je working capitalk/TA, b je  Attr6 je retain earn/TA, c je Attr7 je ebita/TA, d je Attr8 je book equity/TL i e je Attr9 je sales/TA
# z funkci  ce vratit 1 za kompanije u stecaju

z_funkcija<-function(x){
  
  a=x[1]
  b=x[2]
  c=x[3]
  d=x[4]
  e=x[5]
  z=(1.2*a+1.4*b+3.3*c+0.6*d+1*e)
  
  if (z<1.81) {
    
    return(1)# kompanija u stecaju
    
  }else {
    
    return(0) # 0 oznacava kompanije za koje po modelu ne proizilazi stecaj ili se ne moze odrediti
  }
  
}


z_score<-apply(test_noscale[,-1],1,z_funkcija)

z_tab<-table(test_noscale$Stecaj,factor(z_score), dnn = c ("Stvarne","Z_predikcija"))
cm_z1<-confusionMatrix(z_tab,positive = "1")
bs_z1<-BrierScore(as.numeric(test$Stecaj)-1,z_score)


# B emerging market

z_funkcija<-function(x){
  
  a=x[1]
  b=x[2]
  c=x[3]
  d=x[4]
  
  z=(3.25 + 6.56*a+3.26*b+6.72 *c+1.05*d)
  
  if (z<1.1) {
    
    return(1)# kompanija u stecaju
    
  }else {
    
    return(0) # 0 oznacava kompanije za koje po modelu ne proizilazi stecaj ili se ne moze odrediti
  }
  
}

z_score<-apply(test_noscale[, -c(1,6)],1,z_funkcija)

z2_table<-table(test_noscale$Stecaj,factor(z_score),dnn = c ("Stvarne","Z_predikcija"))
cm_z2<-confusionMatrix(z2_table,positive = "1")
bs_z2<-BrierScore(as.numeric(test$Stecaj)-1,z_score)


# Kraj
