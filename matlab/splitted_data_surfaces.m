az_inc_0_180 = readmatrix("splitting_logs_output\az_inc_0_180.txt");
az_inc_180_360 = readmatrix("splitting_logs_output\az_inc_180_360.txt");
el_inc_min_0 = readmatrix("splitting_logs_output\el_inc_min_0.txt");
el_inc_0_max = readmatrix("splitting_logs_output\el_inc_0_max.txt");



x1 = az_inc_0_180(:,1);
y1 = az_inc_0_180(:,2);
z1 = az_inc_0_180(:,3);
% 
% x2 = az_inc_180_360(:,1);
% y2 = az_inc_180_360(:,2);
% z2 = az_inc_180_360(:,3);
% 
% x3 = el_inc_min_0(:,1);
% y3 = el_inc_min_0(:,2);
% z3 = el_inc_min_0(:,3);

x4 = el_inc_0_max(:,1);
y4 = el_inc_0_max(:,2);
z4 = el_inc_0_max(:,3);

curveFitter(x1,y1,z1);
% curveFitter(x2,y2,z2);
% curveFitter(x3,y3,z3);
curveFitter(x4,y4,z4);
% plot3(x1,y1,z1,'o','Color','b','MarkerSize',10,...
%     'MarkerFaceColor','#D9FFFF');
% plot3(x4,y4,z4,'-o','Color','b','MarkerSize',10,...
%     'MarkerFaceColor','#D9FFFF');



