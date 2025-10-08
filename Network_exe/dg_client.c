/* 繧ｳ繝阪け繧ｷ繝ｧ繝ｳ繝ｬ繧ｹ縺ｮ邁｡蜊倥↑繝��繧ｿ讀懃ｴ｢繧ｯ繝ｩ繧､繧｢繝ｳ繝�(dg_client.c) */
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h> /* 繧ｽ繧ｱ繝�ヨ縺ｮ縺溘ａ縺ｮ蝓ｺ譛ｬ逧�↑繝倥ャ繝繝輔ぃ繧､繝ｫ      */
#include <netinet/in.h> /* 繧､繝ｳ繧ｿ繝阪ャ繝医ラ繝｡繧､繝ｳ縺ｮ縺溘ａ縺ｮ繝倥ャ繝繝輔ぃ繧､繝ｫ  */
#include <netdb.h>      /* gethostbyname()繧堤畑縺�ｋ縺溘ａ縺ｮ繝倥ャ繝繝輔ぃ繧､繝ｫ */
#include <unistd.h>
#include <errno.h>
#include <string.h>
#define  MAXHOSTNAME    64
#define	 S_UDP_PORT	(u_short)5000 
#define  MAXKEYLEN	128
#define	 MAXDATALEN	256  
int setup_dgclient(struct hostent*, u_short, struct sockaddr_in*, int*);
void remote_dbsearch(int, struct sockaddr_in*, int);

int main()
{
	int	socd;
	char	s_hostname[MAXHOSTNAME];
	struct hostent	*s_hostent;
	struct sockaddr_in s_address;
	int	s_addrlen;

	/* 繧ｵ繝ｼ繝舌�繝帙せ繝亥錐縺ｮ蜈･蜉� */
	printf("server host name?: "); scanf("%s",s_hostname);
	/* 繧ｵ繝ｼ繝舌�繧ｹ繝医�Internet繧｢繝峨Ξ繧ｹ(繧偵Γ繝ｳ繝舌↓謖√▽hostent讒矩�菴�)繧呈ｱゅａ繧� */
	if((s_hostent = gethostbyname(s_hostname)) == NULL) {
		fprintf(stderr, "server host does not exists\n");
		exit(1);
	}

	/* 繝��繧ｿ繧ｰ繝ｩ繝�繧ｯ繝ｩ繧､繧｢繝ｳ繝医�蛻晄悄險ｭ螳� */
	socd = setup_dgclient(s_hostent, S_UDP_PORT, &s_address, &s_addrlen);

	/* 繝ｪ繝｢繝ｼ繝医ョ繝ｼ繧ｿ繝吶�繧ｹ讀懃ｴ｢ */
	remote_dbsearch(socd, &s_address, s_addrlen);

	close(socd);
	exit(0);
}

int setup_dgclient(struct hostent *hostent, u_short port, struct sockaddr_in *s_addressp, int *s_addrlenp)
{
        int     socd;
        struct sockaddr_in      c_address;
 
	/* 繧､繝ｳ繧ｿ繝ｼ繝阪ャ繝医ラ繝｡繧､繝ｳ縺ｮSOCK_DGRAM(UDP)蝙九た繧ｱ繝�ヨ縺ｮ讒狗ｯ� */
        if((socd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) { perror("socket");exit(1); }
 
	/* 繧ｵ繝ｼ繝舌�繧｢繝峨Ξ繧ｹ(Internet繧｢繝峨Ξ繧ｹ縺ｨ繝昴�繝育分蜿ｷ)縺ｮ菴懈� */
        bzero((char *)s_addressp, sizeof(*s_addressp));
        s_addressp->sin_family = AF_INET;
        s_addressp->sin_port = htons(port);
        bcopy((char *)hostent->h_addr, (char *)&s_addressp->sin_addr, hostent->h_length);
	*s_addrlenp = sizeof(*s_addressp);

	/* 繧ｯ繝ｩ繧､繧｢繝ｳ繝医�繧｢繝峨Ξ繧ｹ(Internet繧｢繝峨Ξ繧ｹ縺ｨ繝昴�繝育分蜿ｷ)縺ｮ菴懈� */
        bzero((char *)&c_address, sizeof(c_address));
        c_address.sin_family = AF_INET;
        c_address.sin_port = htons(0);		       /* 繝昴�繝育分蜿ｷ縺ｮ閾ｪ蜍募牡繧雁ｽ薙※ */
	c_address.sin_addr.s_addr = htonl(INADDR_ANY); /* Internet繧｢繝峨Ξ繧ｹ縺ｮ閾ｪ蜍募牡繧雁ｽ薙※ */

	/* 繧ｯ繝ｩ繧､繧｢繝ｳ繝医い繝峨Ξ繧ｹ縺ｮ繧ｽ繧ｱ繝�ヨ縺ｸ縺ｮ蜑ｲ繧雁ｽ薙※ */
	if(bind(socd, (struct sockaddr *)&c_address, sizeof(c_address)) < 0) { perror("bind");exit(1); }

	return socd;
}

void remote_dbsearch(int socd, struct sockaddr_in *s_addressp, int s_addrlen) /* 繧ｵ繝ｼ繝舌↓繧ｭ繝ｼ繧帝√ｊ讀懃ｴ｢邨先棡(繝��繧ｿ)繧貞女縺大叙繧� */
{
	char	key[MAXKEYLEN+1], data[MAXDATALEN+1];
	int	keylen, datalen;

	/* 繧ｭ繝ｼ繧呈ｨ呎ｺ門�蜉帙°繧牙�蜉� */
	printf("key?: ");scanf("%s",key);
	/* 繧ｭ繝ｼ繧偵た繧ｱ繝�ヨ縺ｫ譖ｸ縺崎ｾｼ繧 */
	keylen = strlen(key);
	if(sendto(socd, key, keylen, 0, (struct sockaddr *)s_addressp, s_addrlen) != keylen) {
		fprintf(stderr, "datagram error\n");
		exit(1);
	}
	/* 讀懃ｴ｢繝��繧ｿ繧偵た繧ｱ繝�ヨ縺九ｉ隱ｭ縺ｿ霎ｼ繧 */
	if((datalen = recvfrom(socd, data, MAXDATALEN, 0, NULL, &s_addrlen)) < 0) { 
		perror("recvfrom");
		exit(1);
	}
	/* 繝��繧ｿ繧呈ｨ呎ｺ門�蜉帙↓蜃ｺ蜉� */
	data[datalen] = '\0';
	fputs("data: ",stdout);puts(data);
}