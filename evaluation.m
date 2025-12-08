clear all;

sequences = 72;

results = {};

fid = fopen('annotations2.txt'); %<--------- FILENAME OF RESULTS GOES HERE
for i=1:sequences
    tmpLine = fgetl(fid);
    tmpArray = strsplit(tmpLine,';');
    results{i,1} = str2num(tmpArray{1,1});
    results{i,2} = tmpArray(1,2:end-1);
end
fclose(fid);

%results = sortrows(results,1);

annotations = {};

%confmat = zeros(19);
confmat = zeros(18);

fid = fopen('annotations.txt'); %<---------- FILENAME OF REFERENCE ANNOTATIONS GOES HERE
for i=1:sequences
    tmpLine = fgetl(fid);
    tmpArray = strsplit(tmpLine,';');
    annotations{i,1} = str2num(tmpArray{1,1});
    annotations{i,2} = tmpArray(1,2:end-1);
end
fclose(fid);

annotations = sortrows(annotations,1);

%numGest = 18;
numGest = 17;
gesturesold = ["ONE" "TWO" "THREE" "FOUR" "OK" "MENU" "POINTING" "LEFT" "RIGHT" "CIRCLE" "V" "CROSS" "GRAB" "PINCH" "TAP" "DENY" "KNOB" "EXPAND"];
gestures = ["ONE" "TWO" "THREE" "FOUR" "OK" "MENU" "LEFT" "RIGHT" "CIRCLE" "V" "CROSS" "GRAB" "PINCH" "TAP" "DENY" "KNOB" "EXPAND"];

jaccardCounts = zeros(numGest,1); 

minOverlapRatio = 0.5;
%classResults columns: Total, Correct, Missed, Misclassified, FalsePositive, Jaccard Index
%Total = Total gestures per class
%Correct = Correctly detected (time window and class)
%Missed = Non-correctly detected
%Misclassified = Detected in a correct time window but with the wrong label
%False Positive = Detected outside any correct time window
classResults = zeros(numGest,6);
%classResults(:,1) = zeros(numGest,1)+16;

for s = 1:sequences
A = annotations{s,2};
R = results{s,2};
if (size(R,2)==0)
    for a = 1:3:size(A,2)
        AA = A(1,a:a+2);
        classResults(find(gestures==AA{1}),1) = classResults(find(gestures==AA{1}),1)+1;
    end
end
found = zeros(1,size(A,2)/3);
for r = 1:3:size(R,2)
    RR = R(1,r:r+2);
    detected = false;
    countA = 0;
    for a = 1:3:size(A,2)

        countA=countA+1;
        AA = A(1,a:a+2);
        if (r==1)
            classResults(find(gestures==AA{1}),1) = classResults(find(gestures==AA{1}),1)+1;
        end
        
        AAlength = str2num(AA{3})-str2num(AA{2});
        %RRlength = str2num(RR{3})-str2num(RR{2});
        overlap = min([str2num(AA{3}) str2num(RR{3})])-max([str2num(AA{2}) str2num(RR{2})]);
        overlapRatio = overlap/AAlength;
        
        %%%%Jaccard Index%%%%
        if (overlap>0 && strcmp(RR{1},AA{1}))
            U = (max([str2num(AA{3}) str2num(RR{3})]))-(min([str2num(AA{2}) str2num(RR{2})]));
            classResults(find(gestures==AA{1}),6) = classResults(find(gestures==AA{1}),6)+ (overlap/U);
            jaccardCounts(find(gestures==AA{1}),1) = jaccardCounts(find(gestures==AA{1}),1)+1;
        end
        
        %%%%%%%%%%%%%%%%%%%%%
        
        if (overlapRatio>minOverlapRatio) %something has been detected where a labeled gesture exists
            detected = true;
            if (strcmp(RR{1},AA{1})) % the lablel matches, it's a correct detection
                if (found(1,countA)~=1) %to avoid counting more than once a duplicated detection 
                    Inx = find(contains(gestures,RR{1}));%predetto
                    Iny = find(contains(gestures,AA{1}));%corretto
                    confmat(Inx,Iny)=confmat(Inx,Iny)+1;
                    found(1,countA) = 1; %marking that a correct dection for this gesture has been found
                    classResults(find(gestures==AA{1}),2) = classResults(find(gestures==AA{1}),2)+1;                    
                else
                    %Inx = 19;%corretto
                    Inx = 18;%corretto
                    Iny = find(contains(gestures,AA{1}));%trovato
                    confmat(Inx,Iny)=confmat(Inx,Iny)+1;
                end
            else %the label doesn't  match, it's a misclassification
                classResults(find(gestures==AA{1}),4) = classResults(find(gestures==AA{1}),4)+1;
                if (found(1,countA)~=1) %to avoid counting more than once a duplicated detection 
                        Inx = find(contains(gestures,RR{1}));%predetto
                        Iny = find(contains(gestures,AA{1}));%corretto
                        confmat(Inx,Iny)=confmat(Inx,Iny)+1;
                else
                    %Inx = 19;%corretto
                    Inx = 18;%corretto
                    Iny = find(contains(gestures,AA{1}));%trovato
                    confmat(Inx,Iny)=confmat(Inx,Iny)+1; 
                end
            end
        end
        if(detected)
            %break;
        end
        
    end
    if (~detected) %the detected gesture window did not match with any of the labeled one, it's a false positive
        classResults(find(gestures==AA{1}),5) = classResults(find(gestures==AA{1}),5)+1;
            %Inx = 19;%corretto
            Inx = 18;%corretto
            Iny = find(contains(gestures,AA{1}));%trovato
            confmat(Inx,Iny)=confmat(Inx,Iny)+1;
    end
    
end
for f = 1:size(found,2)
    if (found(1,f)==0) %this labeled gesture was neither detected correctly or misclassified
        classResults(find(gestures==A{1+((f-1)*3)}),3) = classResults(find(gestures==A{1+((f-1)*3)}),3)+1;
        Inx =find(contains(gestures,A{1+((f-1)*3)}));%corretto
        %Iny = 19;%trovato
        Iny = 18;%trovato

        confmat(Inx,Iny)=confmat(Inx,Iny)+1;
    end
end
end
classResults(:,6) = classResults(:,6)./(jaccardCounts(:,1)+classResults(:,3)+classResults(:,4)+classResults(:,5));

classPrecision = classResults(:,2)./(classResults(:,2)+classResults(:,3)+classResults(:,5));
classRecall = classResults(:,2)./(classResults(:,2)+classResults(:,4));

correctScore = sum(classResults(:,2))/sum(classResults(:,1))
misclassifiedRate = sum(classResults(:,4))/sum(classResults(:,1))
falspositiveRate = sum(classResults(:,5))/sum(classResults(:,1))
%precision = nansum(classPrecision)/numGest
%recall = nansum(classRecall)/numGest

resultsCompact = [classResults(:,2)./16,(classResults(:,4)+classResults(:,5))./16,classResults(:,2)./(classResults(:,2)+classResults(:,3)+classResults(:,4)+classResults(:,5))];
%resultsCompact = [resultsCompact; [sum(resultsCompact(:,1))/18, sum(resultsCompact(:,2))/18, nansum(resultsCompact(:,3))/18]]

resultsCompact = [resultsCompact; [sum(resultsCompact(:,1))/17, sum(resultsCompact(:,2))/17, nansum(resultsCompact(:,3))/17]]
geOLD = ["ONE" "TWO" "THREE" "FOUR" "OK" "MENU" "POINTING" "LEFT" "RIGHT" "CIRCLE" "V" "CROSS" "GRAB" "PINCH" "TAP" "DENY" "KNOB" "EXPAND" "NONGESTURES"];

ge = ["ONE" "TWO" "THREE" "FOUR" "OK" "MENU" "LEFT" "RIGHT" "CIRCLE" "V" "CROSS" "GRAB" "PINCH" "TAP" "DENY" "KNOB" "EXPAND" "NONGESTURES"];


cm = confusionchart(confmat,ge)

sortClasses(cm,ge)

