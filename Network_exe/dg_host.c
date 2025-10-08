/* 繧ｳ繝阪け繧ｷ繝ｧ繝ｳ繝ｬ繧ｹ縺ｮ邁｡蜊倥↑繝��繧ｿ讀懃ｴ｢繧ｵ繝ｼ繝�(dg_server.c) */
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h> /* 繧ｽ繧ｱ繝�ヨ縺ｮ縺溘ａ縺ｮ蝓ｺ譛ｬ逧�↑繝倥ャ繝繝輔ぃ繧､繝ｫ      */
#include <netinet/in.h> /* 繧､繝ｳ繧ｿ繝阪ャ繝医ラ繝｡繧､繝ｳ縺ｮ縺溘ａ縺ｮ繝倥ャ繝繝輔ぃ繧､繝ｫ  */
#include <netdb.h>      /* gethostbyname()繧堤畑縺�ｋ縺溘ａ縺ｮ繝倥ャ繝繝輔ぃ繧､繝ｫ */
#include <unistd.h>     /* gethostname()繧堤畑縺�ｋ縺溘ａ縺ｮ繝倥ャ繝繝輔ぃ繧､繝ｫ */
#include <errno.h>
#include <string.h>
#define  MAXHOSTNAME	64
#define  S_UDP_PORT	(u_short)5000  /* 譛ｬ繧ｵ繝ｼ繝舌′逕ｨ縺�ｋ繝昴�繝育分蜿ｷ */
#define  MAXKEYLEN	128
#define  MAXDATALEN	256
int setup_dgserver(struct hostent*, u_short);
void db_search(int);

int main()
{
	int	socd;
	char	s_hostname[MAXHOSTNAME];
	struct hostent	*s_hostent;

	/* 繧ｵ繝ｼ繝舌�繝帙せ繝亥錐縺ｨ縺昴�Internet繧｢繝峨Ξ繧ｹ(繧偵Γ繝ｳ繝舌↓謖√▽hostent讒矩�菴�)繧呈ｱゅａ繧� */
	gethostname(s_hostname, sizeof(s_hostname));
	s_hostent = gethostbyname(s_hostname);

	/* 繝��繧ｿ繧ｰ繝ｩ繝�繧ｵ繝ｼ繝舌�蛻晄悄險ｭ螳� */
	socd = setup_dgserver(s_hostent, S_UDP_PORT);

	/* 繧ｯ繝ｩ繧､繧｢繝ｳ繝医°繧峨�繝��繧ｿ讀懃ｴ｢隕∵ｱゅ�蜃ｦ逅� */
	db_search(socd);
	return 0;
}

int setup_dgserver(struct hostent *hostent, u_short port)
{
	int	socd;
	struct sockaddr_in	s_address;

	/* 繧､繝ｳ繧ｿ繝ｼ繝阪ャ繝医ラ繝｡繧､繝ｳ縺ｮSOCK_DGRAM(UDP)蝙九た繧ｱ繝�ヨ縺ｮ讒狗ｯ� */
	if((socd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) { perror("socket");exit(1); }

	/* 繧｢繝峨Ξ繧ｹ(Internet繧｢繝峨Ξ繧ｹ縺ｨ繝昴�繝育分蜿ｷ)縺ｮ菴懈� */
	bzero((char *)&s_address, sizeof(s_address));
	s_address.sin_family = AF_INET;
	s_address.sin_port = htons(port);
	bcopy((char *)hostent->h_addr, (char *)&s_address.sin_addr, hostent->h_length);

	/* 繧｢繝峨Ξ繧ｹ縺ｮ繧ｽ繧ｱ繝�ヨ縺ｸ縺ｮ蜑ｲ繧雁ｽ薙※ */
	if(bind(socd, (struct sockaddr *)&s_address, sizeof(s_address)) < 0) { perror("bind");exit(1); }

	return socd;
}

void db_search(int socd) /* 繧ｯ繝ｩ繧､繧｢繝ｳ繝医′繝��繧ｿ讀懃ｴ｢隕∵ｱゅｒ蜃ｦ逅�☆繧� */
{
	struct sockaddr_in	c_address;
	int	c_addrlen;
	char	key[MAXKEYLEN+1], data[MAXDATALEN+1];
	int	keylen, datalen;
	static char *db[] = {"amano-taro","0426-91-9418","ishida-jiro","0426-91-9872",
                             "ueda-saburo","0426-91-9265","ema-shiro","0426-91-7254",
                             "ooishi-goro","0426-91-9618",NULL};
	char	**dbp;

	while(1) {
		/* 繧ｭ繝ｼ繧偵た繧ｱ繝�ヨ縺九ｉ隱ｭ縺ｿ霎ｼ繧 */
		c_addrlen = sizeof(c_address);
		if((keylen = recvfrom(socd, key, MAXKEYLEN, 0, (struct sockaddr *)&c_address, &c_addrlen)) < 0) {
			perror("recvfrom");
			exit(1);
		}
		key[keylen] = '\0';
		printf("Received key> %s\n",key);
		/* 繧ｭ繝ｼ繧堤畑縺�※繝��繧ｿ讀懃ｴ｢ */
		dbp = db;
		while(*dbp) {
			if(strcmp(key, *dbp) == 0) {
				strcpy(data, *(++dbp));
				break;
			}
			dbp += 2;
		}
		if(*dbp == NULL) strcpy(data, "No entry");
	
		/* 讀懃ｴ｢縺励◆繝��繧ｿ繧偵た繧ｱ繝�ヨ縺ｫ譖ｸ縺崎ｾｼ繧 */
		datalen = strlen(data);
		if(sendto(socd, data, datalen, 0, (struct sockaddr *)&c_address, c_addrlen) != datalen) {
			fprintf(stderr, "datagram error\n"); 
			exit(1);
		}
		printf("Sent data> %s\n", data);
	}
}