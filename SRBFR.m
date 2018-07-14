function acc=SRBFR(n,f)
 n=n;
 filepath=f;
 filename=dir(filepath);
 filenum=length(filename);
 seq=1;
 seqi=1;
 seqt=1;
 imsum=[];
 warning off
 for i=1:filenum
    path=filename(i);
    imagepath=dir(fullfile(filepath,path.name,'*.pgm'));
    imagenum=length(imagepath);
    if imagenum>0
        k=randperm(imagenum);
        user(seq).pa=path;
        seq=seq+1;
        for jf=1:n
            imagename=imagepath(k(jf)).name;
            if ~isempty(strfind(imagename,'Ambient'))
                imagename=imagepath(k(n+1)).name;
                image=imread(fullfile(filepath,path.name,imagename));
                ima=imresize(image,[50 50]);
                ima=im2double(ima);
                [imheight,imwidth]=size(ima);
                imvec=reshape(ima,imheight*imwidth,1);
                user(seqi).image=imvec;
                user(seqi).ipathname=path.name;
                seqi=seqi+1;
                imsum=[imsum imvec];
                continue
            end
            image=imread(fullfile(filepath,path.name,imagename));
            ima=imresize(image,[50 50]);
            ima=im2double(ima);
            [imheight,imwidth]=size(ima);
            imvec=reshape(ima,imheight*imwidth,1);
            user(seqi).image=imvec;
            user(seqi).ipathname=path.name;
            seqi=seqi+1;
            imsum=[imsum imvec];
        end
        for m=n:imagenum
            testname=imagepath(k(m)).name;
            if ~isempty(strfind(testname,'Ambient'))
                try
                 testname=imagepath(k(m+1)).name;
                 test=imread(fullfile(filepath,path.name,testname));
                 testa=imresize(test,[50 50]);
                 testa=im2double(testa);
                 [imheight,imwidth]=size(testa);
                 testv=reshape(testa,imheight*imwidth,1);
                 user(seqt).test=testv;
                 user(seqt).pathname=path.name;
                 seqt=seqt+1;
                catch
                    continue
                end
             continue
            end
            test=imread(fullfile(filepath,path.name,testname));
            testa=imresize(test,[50 50]);
            testa=im2double(testa);
            [imheight,imwidth]=size(testa);
            testv=reshape(testa,imheight*imwidth,1);
            user(seqt).test=testv;
            user(seqt).pathname=path.name;
            seqt=seqt+1;
        end
    end
 end
 imsumt=imsum';
 [coeff,scores,latent]=princomp(imsumt);
 lanum=length(latent);
 sumla=0;
 bc=0;
 while sumla<0.95
    bc=bc+1;
    sumlat=sum(sum(latent));
    lat=latent/sumlat;
    sumla=sumla+lat(bc);
 end
 change=coeff\eye(2500);
 score=[];
 score=scores(:,1:bc);
 zz=0;
 for l=1:seqt-1%In fact,it should be 1:seqt-1,but it cost too much time
    observ=getfield(user,{l},'test');
    obs=change*observ;
    obs1=[];
    obs1=obs(1:bc,:);
    xini=zeros(38*n,1);
    qw=[];
    qi=[];
    try
       [x]=feature_sign(score',obs1,0.0006,xini);
       xnum=length(x);
       t=1;
       for i=1:xnum
          if x(i)>0.2
             qi(t)=i;
             init=getfield(user,{i},'image');
             err=norm(init-observ);
             qw(t)=err;
             t=t+1;
          end
       end
       [~,aaa]=max(x);
       %weizhi=qi(aaa);
       result=getfield(user,{aaa},'ipathname');
       realre=getfield(user,{l},'pathname');
       if strcmpi(realre,result)==1
          zz=zz+1;
       end
    catch
        continue;
    end
 end
 acc=zz/(seqt-1);

 function [x]=feature_sign(B,y,lambda,init_x)
%%%%%%min_x 0.5\|y-Bx\|_2^2+lambda\|x\|_1

 nbases=size(B,2);

 OptTol = 1e-5;

 if nargin < 4,
    x=zeros(nbases, 1);
 else
    x = init_x;
 end;

 theta=sign(x);          %sign flag
 a=(x~=0);               %active set

 optc=0;

 By=B'*y;
 B_h=B(:,a);
 x_h=x(a);
 Bx_h=B_h*x_h;
 all_d=2*(B'*Bx_h-By);
 [ma mi]=max(abs(all_d).*(~a));

 while optc==0,

    optc=1;

    if all_d(mi)>lambda+1e-10,
        theta(mi)=-1;
        a(mi)=1;
        b=B(:,mi);
        x(mi)=(lambda-all_d(mi))/(b'*b*2);
    elseif all_d(mi)<-lambda-1e-10,
        theta(mi)=1;
        a(mi)=1;
        b=B(:,mi);
        x(mi)=(-lambda-all_d(mi))/(b'*b*2);
    else
        if sum(a)==0,
            lambda=ma-2*1e-10;
            optc=0;
            b=B(:,mi);
            x(mi)=By(mi)/(b'*b);
            break;
        end
    end

    opts=0;
    B_h=B(:,a);
    x_h=x(a);
    theta_h=theta(a);

    while opts==0,
        opts=1;

        if size(B_h,2)<=length(y),
            BB=B_h'*B_h;
            x_new=BB\(B_h'*y-lambda*theta_h/2);
            o_new=L1_cost(y,B_h,x_new,lambda);

            %cost based on changing sign
            s=find(sign(x_new)~=theta_h);
            x_min=x_new;
            o_min=o_new;
            for j=1:length(s),
                zd=s(j);
                x_s=x_h-x_h(zd)*(x_new-x_h)/(x_new(zd)-x_h(zd));
                x_s(zd)=0;  %make sure it's zero
                o_s=L1_cost(y,B_h,x_s,lambda);
                if o_s<o_min,
                    x_min=x_s;
                    o_min=o_s;
                end
            end
        else
            d=x_h-B_h'*((B_h*B_h')\(B_h*x_h));
            q=x_h./(d+eps);
            x_min=x_h;
            o_min=L1_cost(y,B_h,x_h,lambda);
            for j=1:length(q),
                zd=q(j);
                x_s=x_h-zd*d;
                x_s(j)=0;       %make sure it's zero
                o_s=L1_cost(y,B_h,x_s,lambda);
                if o_s<o_min,
                   x_min=x_s;
                   o_min=o_s;
                end
            end
        end

        x(a)=x_min;

        a=(x~=0);
        theta=sign(x);

        B_h=B(:,a);
        x_h=x(a);
        theta_h=theta(a);
        Bx_h=B_h*x_h;

        active_d=2*(B_h'*(Bx_h-y))+lambda*theta_h;

        if ~isempty(find(abs(active_d)>OptTol)),
            opts=0;
        end
    end

    all_d=2*(B'*Bx_h-By);

    [ma mi]=max(abs(all_d).*(~a));
    if ma>lambda+OptTol,
        optc=0;
    end
 end

 return;
 end

 function cost=L1_cost(y,B,x,lambda)
    tmp = y-B*x;
    cost = tmp'*tmp+lambda*norm(x,1);

 return
 end
warning on
end
